cmake_minimum_required(VERSION 3.16)

project(Hello)

add_executable(Hello Hello.cpp)

install(TARGETS Hello)

enable_testing()
add_test(NAME Hello COMMAND Hello)
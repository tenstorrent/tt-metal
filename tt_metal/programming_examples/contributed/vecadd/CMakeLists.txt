cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_vecadd)

# This is a test to see if the vecadd example is working
# This cmakelist file will never actually be used in CI.

add_executable(metal_example_vecadd)
target_sources(metal_example_vecadd PRIVATE vecadd.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_vecadd PUBLIC TT::Metalium)

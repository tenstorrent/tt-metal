cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_loopback)

add_executable(metal_example_loopback)
target_sources(metal_example_loopback PRIVATE loopback.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_loopback PUBLIC TT::Metalium)

cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_pad)

add_executable(metal_example_pad)
target_sources(metal_example_pad PRIVATE pad.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_pad PUBLIC TT::Metalium) 
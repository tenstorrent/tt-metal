cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_eltwise_binary)

add_executable(metal_example_eltwise_binary)
target_sources(metal_example_eltwise_binary PRIVATE eltwise_binary.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_eltwise_binary PUBLIC TT::Metalium)

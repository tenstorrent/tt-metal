cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_eltwise_ternary)

add_executable(metal_example_eltwise_ternary)
target_sources(metal_example_eltwise_ternary PRIVATE eltwise_ternary.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_eltwise_ternary PUBLIC TT::Metalium)

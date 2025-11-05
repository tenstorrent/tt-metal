cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_custom_smoothstep)

add_executable(metal_example_custom_smoothstep)
target_sources(metal_example_custom_smoothstep PRIVATE custom_sfpi_smoothstep.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_custom_smoothstep PUBLIC TT::Metalium)

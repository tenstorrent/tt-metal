cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_custom_sfpi_add)

add_executable(metal_example_custom_sfpi_add)
target_sources(metal_example_custom_sfpi_add PRIVATE custom_sfpi_add.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_custom_sfpi_add PUBLIC TT::Metalium)

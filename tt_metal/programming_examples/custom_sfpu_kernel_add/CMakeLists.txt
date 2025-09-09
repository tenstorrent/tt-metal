cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_custom_sfpu_kernel_add)

add_executable(metal_example_custom_sfpu_kernel_add)
target_sources(metal_example_custom_sfpu_kernel_add PRIVATE custom_sfpu_kernel_add.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_custom_sfpu_kernel_add PUBLIC TT::Metalium)

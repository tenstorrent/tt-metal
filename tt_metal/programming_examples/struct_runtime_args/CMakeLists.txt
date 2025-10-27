cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_struct_runtime_args)

add_executable(metal_example_struct_runtime_args)
target_sources(metal_example_struct_runtime_args PRIVATE struct_runtime_args.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_struct_runtime_args PUBLIC TT::Metalium)

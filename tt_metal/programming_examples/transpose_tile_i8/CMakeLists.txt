cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_transpose_tile_i8)

add_executable(metal_example_transpose_tile_i8)
target_sources(metal_example_transpose_tile_i8 PRIVATE transpose_tile_i8.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_transpose_tile_i8 PUBLIC TT::Metalium)

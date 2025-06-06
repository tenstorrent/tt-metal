cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_matmul_multicore_reuse)

add_executable(metal_example_matmul_multicore_reuse)
target_sources(metal_example_matmul_multicore_reuse PRIVATE matmul_multicore_reuse.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_matmul_multicore_reuse PUBLIC TT::Metalium)

target_include_directories(metal_example_matmul_multicore_reuse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

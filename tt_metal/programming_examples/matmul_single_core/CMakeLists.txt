cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_matmul_single_core)

add_executable(metal_example_matmul_single_core)
target_sources(metal_example_matmul_single_core PRIVATE matmul_single_core.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_matmul_single_core PUBLIC TT::Metalium)

target_include_directories(metal_example_matmul_single_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
) 
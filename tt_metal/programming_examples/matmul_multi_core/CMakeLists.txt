cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_matmul_multi_core)

add_executable(metal_example_matmul_multi_core)
target_sources(metal_example_matmul_multi_core PRIVATE matmul_multi_core.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_matmul_multi_core PUBLIC TT::Metalium) 
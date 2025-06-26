cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_matmul)

add_subdirectory(matmul_common)

add_subdirectory(matmul_single_core)
add_subdirectory(matmul_multi_core)
add_subdirectory(matmul_multicore_reuse)

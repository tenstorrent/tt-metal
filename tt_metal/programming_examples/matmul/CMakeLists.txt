cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_matmul)

add_subdirectory(matmul_common)

add_executable(metal_example_matmul_single_core)
target_sources(metal_example_matmul_single_core PRIVATE matmul_single_core/matmul_single_core.cpp)

add_executable(metal_example_matmul_multi_core)
target_sources(metal_example_matmul_multi_core PRIVATE matmul_multi_core/matmul_multi_core.cpp)

add_executable(metal_example_matmul_multicore_reuse)
target_sources(metal_example_matmul_multicore_reuse PRIVATE matmul_multicore_reuse/matmul_multicore_reuse.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(
    metal_example_matmul_single_core
    PUBLIC
        TT::Metalium
        Matmul::Common
)
target_link_libraries(
    metal_example_matmul_multi_core
    PUBLIC
        TT::Metalium
        Matmul::Common
)
target_link_libraries(
    metal_example_matmul_multicore_reuse
    PUBLIC
        TT::Metalium
        Matmul::Common
)

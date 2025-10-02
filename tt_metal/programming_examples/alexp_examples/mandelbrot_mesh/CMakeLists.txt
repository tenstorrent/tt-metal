# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22...3.30)
project(mandelbrot_mesh)

# Mandelbrot Mesh - Fixed-point version (recommended)
add_executable(mandelbrot_mesh mandelbrot_mesh.cpp)
target_link_libraries(
    mandelbrot_mesh
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
target_include_directories(mandelbrot_mesh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mandelbrot_mesh PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

# Mandelbrot Mesh - Simple version with detailed comments
add_executable(mandelbrot_mesh_simple mandelbrot_mesh_simple.cpp)
target_link_libraries(
    mandelbrot_mesh_simple
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
target_include_directories(mandelbrot_mesh_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mandelbrot_mesh_simple PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

# Multi-Core Mandelbrot Mesh Implementation
add_executable(mandelbrot_multi_core_mesh mandelbrot_multi_core.cpp)
target_link_libraries(mandelbrot_multi_core_mesh tt_metal)
target_include_directories(mandelbrot_multi_core_mesh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mandelbrot_multi_core_mesh PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

# Multi-Core Mandelbrot Mesh with Performance Benchmarking
add_executable(mandelbrot_multi_core_benchmark mandelbrot_multi_core_benchmark.cpp)
target_link_libraries(mandelbrot_multi_core_benchmark tt_metal)
target_include_directories(mandelbrot_multi_core_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mandelbrot_multi_core_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

# Comprehensive Benchmark Tool for All Implementations
add_executable(benchmark_all_implementations benchmark_all_implementations.cpp)
target_link_libraries(benchmark_all_implementations tt_metal)
target_include_directories(benchmark_all_implementations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(benchmark_all_implementations PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

# Simple Performance Measurement Example
add_executable(simple_performance_example simple_performance_example.cpp)
target_link_libraries(simple_performance_example tt_metal)
target_include_directories(simple_performance_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(simple_performance_example PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)

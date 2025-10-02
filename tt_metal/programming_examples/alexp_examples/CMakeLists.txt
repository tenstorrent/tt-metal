cmake_minimum_required(VERSION 3.22...3.30)
project(alexp_examples)

# Aggregate CMake for AlexP examples
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nlp_concat_heads_boltz_example)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_mesh)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/e4)

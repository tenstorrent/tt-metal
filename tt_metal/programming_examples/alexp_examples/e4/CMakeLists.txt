cmake_minimum_required(VERSION 3.22...3.30)
project(alexp_e4_examples)

# E4 examples subdirectories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/elementwise_add)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/multi_tensix_elementwise_add)

cmake_minimum_required(VERSION 3.22...3.30)
project(metal_examples_matmul)

# Define source files for each example
set(MATMUL_SINGLE_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core/matmul_single_core.cpp)
set(MATMUL_MULTI_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core/matmul_multi_core.cpp)
set(MATMUL_MULTICORE_REUSE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse/matmul_multicore_reuse.cpp)
set(MATMUL_MULTICORE_REUSE_MCAST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp
)

# Create executables using the macro
CREATE_PGM_EXAMPLES_EXE("${MATMUL_SINGLE_CORE_SRCS}" "matmul") # output binaries to build/programming_examples/matmul
CREATE_PGM_EXAMPLES_EXE("${MATMUL_MULTI_CORE_SRCS}" "matmul")
CREATE_PGM_EXAMPLES_EXE("${MATMUL_MULTICORE_REUSE_SRCS}" "matmul")
CREATE_PGM_EXAMPLES_EXE("${MATMUL_MULTICORE_REUSE_MCAST_SRCS}" "matmul")

# Include the common directory for shared code
include_directories(matmul_common)

# Create a custom target that depends on all the test targets
add_custom_target(matmul DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})


set(PROGRAMMING_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute/add_2_integers_in_compute
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_riscv/add_2_integers_in_riscv
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_binary/eltwise_binary
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu/eltwise_sfpu
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_compute_kernel/hello_world_compute_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datamovement_kernel/hello_world_datamovement_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datatypes_kernel/hello_world_datatypes_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/loopback/loopback
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core/matmul_multi_core
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse/matmul_multicore_reuse
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core/matmul_single_core
)

include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

CREATE_PGM_EXAMPLES_EXE("${PROGRAMMING_EXAMPLES_SRCS}" "")  # no subdir, output binaries straight to build/programming_examples

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/contributed)

add_custom_target(programming_examples DEPENDS profiler_examples contributed ${PROGRAMMING_EXAMPLES_TEST_TARGETS})

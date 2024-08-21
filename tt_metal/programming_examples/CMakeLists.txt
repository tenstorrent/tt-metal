
set(PROGRAMMING_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute/add_2_integers_in_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_binary/eltwise_binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu/eltwise_sfpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_compute_kernel/hello_world_compute_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datatypes_kernel/hello_world_datatypes_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loopback/loopback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core/matmul_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse/matmul_multicore_reuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core/matmul_single_core.cpp
)

include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

CREATE_PGM_EXAMPLES_EXE("${PROGRAMMING_EXAMPLES_SRCS}" "")  # no subdir, output binaries straight to build/programming_examples

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/contributed)

add_custom_target(programming_examples DEPENDS profiler_examples contributed ${PROGRAMMING_EXAMPLES_TEST_TARGETS})

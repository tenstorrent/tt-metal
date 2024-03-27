
set(PERF_MICROBENCH_TESTS_SRCS
    dispatch/test_pgm_dispatch
    dispatch/test_bw_and_latency
    dispatch/test_dispatcher
    dispatch/test_prefetcher
    ethernet/test_ethernet_read_and_send_data
    ethernet/test_workers_and_erisc_datamover_unidirectional
    noc/test_noc_unicast_vs_multicast_to_single_core_latency
    old/matmul/matmul_global_l1
    old/matmul/matmul_local_l1
    old/noc/test_noc_read_global_l1
    old/noc/test_noc_read_local_l1
    old/pcie/test_enqueue_rw_buffer
    old/pcie/test_rw_buffer
    old/pcie/test_rw_device_dram
    old/pcie/test_rw_device_l1
    1_compute_mm/test_compute_mm
    2_noc_adjacent/test_noc_adjacent
    2_noc_rtor/test_noc_rtor
    3_pcie_transfer/test_rw_buffer
    6_dram_offchip/test_dram_offchip
    7_kernel_launch/test_kernel_launch
)

foreach (TEST ${PERF_MICROBENCH_TESTS_SRCS})
    get_filename_component(TEST_TARGET ${TEST} NAME)
    get_filename_component(TEST_DIR ${TEST} DIRECTORY)
    # test_rw_buffer have two versions >:/ can we remove one?
    if (${TEST} STREQUAL "old/pcie/test_rw_buffer")
        set(TEST_TARGET "test_rw_buffer_old")
    endif()

    add_executable(${TEST_TARGET} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST})
    target_link_libraries(${TEST_TARGET} PUBLIC tt_metal_libs stdc++fs yaml-cpp gtest gtest_main m)
    target_include_directories(${TEST_TARGET} PRIVATE
        ${UMD_HOME}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tt_metal
        ${CMAKE_SOURCE_DIR}/tt_eager
        ${CMAKE_SOURCE_DIR}/tt_metal/common
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal/perf_microbenchmark/${TEST_DIR})
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND PERF_MICROBENCH_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(metal_perf_microbenchmark_tests DEPENDS ${PERF_MICROBENCH_TEST_TARGETS})

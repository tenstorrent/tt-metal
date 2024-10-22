
set(PERF_MICROBENCH_TESTS_SRCS
    dispatch/test_pgm_dispatch.cpp
    dispatch/test_bw_and_latency.cpp
    dispatch/test_dispatcher.cpp
    dispatch/test_prefetcher.cpp
    ethernet/test_ethernet_read_and_send_data.cpp
    ethernet/test_workers_and_erisc_datamover_unidirectional.cpp
    ethernet/test_ethernet_bidirectional_bandwidth_no_edm.cpp
    ethernet/test_ethernet_link_ping_latency_no_edm.cpp
    ethernet/test_ethernet_hop_latencies_no_edm.cpp
    routing/test_tx_rx.cpp
    routing/test_mux_demux.cpp
    routing/test_vc_mux_demux.cpp
    routing/test_mux_demux_2level.cpp
    routing/test_tunnel_1cq.cpp
    routing/test_tunnel_2cq.cpp
    routing/test_uni_tunnel.cpp
    routing/test_vc_uni_tunnel.cpp
    routing/test_uni_tunnel_single_chip.cpp
    routing/test_bi_tunnel.cpp
    routing/test_vc_bi_tunnel.cpp
    noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp
    old/matmul/matmul_global_l1.cpp
    old/matmul/matmul_local_l1.cpp
    old/noc/test_noc_read_global_l1.cpp
    old/noc/test_noc_read_local_l1.cpp
    old/pcie/test_enqueue_rw_buffer.cpp
    old/pcie/test_rw_buffer.cpp
    old/pcie/test_rw_device_dram.cpp
    old/pcie/test_rw_device_l1.cpp
    1_compute_mm/test_compute_mm.cpp
    2_noc_adjacent/test_noc_adjacent.cpp
    2_noc_rtor/test_noc_rtor.cpp
    3_pcie_transfer/test_rw_buffer.cpp
    3_pcie_transfer/test_pull_from_pcie.cpp
    6_dram_offchip/test_dram_offchip.cpp
    7_kernel_launch/test_kernel_launch.cpp
    8_dram_adjacent_core_read/test_dram_read.cpp
    9_dram_adjacent_read_remote_l1_write/test_dram_read_l1_write.cpp
    10_dram_read_remote_cb_sync/test_dram_read_remote_cb.cpp
)

foreach (TEST_SRC ${PERF_MICROBENCH_TESTS_SRCS})
    get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
    # test_rw_buffer have two versions >:/ can we remove one?
    if (${TEST_SRC} STREQUAL "old/pcie/test_rw_buffer.cpp")
        set(TEST_TARGET "test_rw_buffer_old")
    endif()

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_link_libraries(${TEST_TARGET} PUBLIC test_metal_common_libs)
    if (${TEST_TARGET} MATCHES "test_dram_read")
        target_link_libraries(${TEST_TARGET} PRIVATE yaml-cpp::yaml-cpp)
    endif()

    target_include_directories(${TEST_TARGET} PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/ttnn/cpp/ttnn/deprecated # this all should go away and be replaced with link to ttnn
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal/perf_microbenchmark/${TEST_DIR})
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND PERF_MICROBENCH_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(metal_perf_microbenchmark_tests DEPENDS ${PERF_MICROBENCH_TEST_TARGETS})

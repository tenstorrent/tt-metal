# Source files for tt_metal perf_microbenchmark tests
# Module owners should update this file when adding/removing/renaming source files

set(PERF_MICROBENCH_TESTS_SRCS
    dispatch/test_pgm_dispatch.cpp
    dispatch/benchmark_rw_buffer.cpp
    ethernet/test_ethernet_bidirectional_bandwidth_no_edm.cpp
    ethernet/test_ethernet_link_ping_latency_no_edm.cpp
    ethernet/test_all_ethernet_links.cpp
    routing/test_tt_fabric_mux_bandwidth.cpp
    routing/test_tt_fabric.cpp
    noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp
    tensix/test_gathering.cpp
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
    6_dram_offchip/test_dram_offchip.cpp
    7_kernel_launch/test_kernel_launch.cpp
    8_dram_adjacent_core_read/test_dram_read.cpp
    9_dram_adjacent_read_remote_l1_write/test_dram_read_l1_write.cpp
    10_dram_read_remote_cb_sync/test_dram_read_remote_cb.cpp
    11_remote_cb_sync_matmul_single_core/test_remote_cb_sync_matmul.cpp
)

set(X86_64_ONLY_TESTS
    dispatch/test_prefetcher.cpp
    dispatch/test_bw_and_latency.cpp
    dispatch/test_dispatcher.cpp
    3_pcie_transfer/test_pull_from_pcie.cpp
)

set(TEST_TT_FABRIC_ADDITIONAL_SOURCES
    routing/tt_fabric_test_common_types.cpp
    routing/tt_fabric_test_config.cpp
    routing/tt_fabric_test_results.cpp
    routing/tt_fabric_test_latency_results.cpp
    routing/tt_fabric_test_bandwidth_results.cpp
    routing/tt_fabric_test_context.cpp
    routing/tt_fabric_test_bandwidth_profiler.cpp
    routing/tt_fabric_test_device_setup.cpp
    routing/tt_fabric_test_progress_monitor.cpp
    routing/tt_fabric_test_eth_readback.cpp
    routing/tt_fabric_test_code_profiler.cpp
    routing/tt_fabric_telemetry_manager.cpp
)

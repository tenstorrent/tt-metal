set(PERF_MICROBENCH_TESTS_SRCS
    dispatch/test_pgm_dispatch.cpp
    dispatch/benchmark_rw_buffer.cpp
    ethernet/test_ethernet_read_and_send_data.cpp
    ethernet/test_workers_and_erisc_datamover_unidirectional.cpp
    ethernet/test_ethernet_bidirectional_bandwidth_no_edm.cpp
    ethernet/test_ethernet_link_ping_latency_no_edm.cpp
    ethernet/test_all_ethernet_links.cpp
    ethernet/test_ethernet_hop_latencies_no_edm.cpp
    routing/test_tt_fabric_sanity.cpp
    routing/test_tt_fabric_mux_bandwidth.cpp
    routing/test_tt_fabric_multi_hop_sanity.cpp
    routing/test_tt_fabric_socket_sanity.cpp
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

# These tests use SIMD intrinsics and are not portable.
set(X86_64_ONLY_TESTS
    dispatch/test_prefetcher.cpp
    dispatch/test_bw_and_latency.cpp
    dispatch/test_dispatcher.cpp
    3_pcie_transfer/test_pull_from_pcie.cpp
)

# Add x86_64 tests only on appropriate platforms.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND PERF_MICROBENCH_TESTS_SRCS ${X86_64_ONLY_TESTS})
endif()

foreach(TEST_SRC ${PERF_MICROBENCH_TESTS_SRCS})
    get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
    # test_rw_buffer have two versions >:/ can we remove one?
    if(${TEST_SRC} STREQUAL "old/pcie/test_rw_buffer.cpp")
        set(TEST_TARGET "test_rw_buffer_old")
    endif()
    string(REPLACE "wormhole" "wormhole_b0" TEST_TARGET ${TEST_TARGET})

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_link_libraries(${TEST_TARGET} PRIVATE test_metal_common_libs)

    if(${TEST_SRC} STREQUAL "dispatch/test_pgm_dispatch.cpp")
        target_link_libraries(${TEST_TARGET} PRIVATE benchmark::benchmark)
    endif()
    if(${TEST_SRC} STREQUAL "dispatch/benchmark_rw_buffer.cpp")
        target_link_libraries(${TEST_TARGET} PRIVATE benchmark::benchmark)
    endif()

    if(${TEST_SRC} STREQUAL "routing/test_tt_fabric.cpp")
        target_link_libraries(${TEST_TARGET} PRIVATE yaml-cpp::yaml-cpp)
    endif()
    target_include_directories(
        ${TEST_TARGET}
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/ttnn/cpp/ttnn/deprecated # this all should go away and be replaced with link to ttnn
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/perf_microbenchmark/${TEST_DIR}
    )
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND PERF_MICROBENCH_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(metal_perf_microbenchmark_tests DEPENDS ${PERF_MICROBENCH_TEST_TARGETS})

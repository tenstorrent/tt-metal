
set(UNIT_TESTS_COMMON_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/test_device_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/test_bit_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/test_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/test_flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_large_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_multi_core_multi_dram_inX_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_multi_core_X_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/test_matmul_X_tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_eth_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_invalid_print_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_print_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_all_harts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_before_finish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_hanging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_raise_wait.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram/test_dram_to_l1_multicast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram/test_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize_delays.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_pause.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_ringbuf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_waypoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_link_training.cpp
)
add_library(unit_tests_common_o OBJECT ${UNIT_TESTS_COMMON_SRC})
target_link_libraries(unit_tests_common_o PUBLIC compiler_flags metal_header_directories gtest gtest_main magic_enum fmt)
target_include_directories(unit_tests_common_o PUBLIC
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt
    ${PROJECT_SOURCE_DIR}/tt_metal/common
    ${PROJECT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

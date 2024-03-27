
set(UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/test_free_list_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/test_l1_banking_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/initialize_semaphores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/runtime_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/test_noc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic/test_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/test_banked.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/test_buffer_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/test_sharded_l1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/test_simple_dram_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/test_simple_l1_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer/test_CircularBuffer_allocation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer/test_CircularBuffer_creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/binary/single_core_binary_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/matmul/single_core_matmul_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/sfpu/sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_adjacent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_contains.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_intersects.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_merge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_construct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_merge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram/direct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_apis/test_tilize_untilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/basic_eth_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/buffer_movement_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/device_cluster_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/erisc_app_direct_send.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/ring_gather_kernels.cpp
)

add_executable(unit_tests ${UNIT_TESTS_SRC} $<TARGET_OBJECTS:unit_tests_common_o>)

target_link_libraries(unit_tests PUBLIC tt_metal_libs stdc++fs yaml-cpp gtest gtest_main m)
target_include_directories(unit_tests PRIVATE
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer
)
set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal)



set (TT_METAL_TESTS
    test_bmm
    test_add_two_ints
    # test_compile_args             <- not tested in run_tt_metal.py
    # test_eltwise_binary           <- fast dispatch
    # test_eltwise_unary            <- not tested in run_tt_metal.py
    test_matmul_single_tile_bfp8b
    test_matmul_single_tile_output_in_l1
    test_dram_loopback_single_core
    test_datacopy_bfp8b
    test_datacopy
    test_datacopy_output_in_l1
    test_dataflow_cb
    test_transpose_hc
    test_transpose_wh
    test_multiple_programs
    test_multi_core_kernel
    test_unpack_tilize
    test_unpack_untilize
    test_interleaved_layouts
    test_interleaved_l1_buffer
    test_bcast
    test_generic_binary_reader_matmul_large_block
    # test_3x3conv_as_matmul_large_block        <- not tested in run_tt_metal.py
    # test_l1_to_l1_multi_core                    <- test borked
    test_dram_copy_sticks_multi_core
    test_reduce_h
    test_reduce_w
    test_reduce_hw
    test_untilize_eltwise_binary
    test_bfp8_conversion
    # test_bfp4_conversion            <- not tested in run_tt_metal.py
    test_core_range_set
    test_compile_sets_kernel_binaries
    test_compile_program
)

foreach (TEST ${TT_METAL_TESTS})
    add_executable(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST})
    target_link_libraries(${TEST} PUBLIC ${COMMON_LIBS} tt_metal_libs)
    target_link_libraries(${TEST} PUBLIC stdc++fs yaml-cpp gtest gtest_main m)
    target_include_directories(${TEST} PRIVATE
        ${UMD_HOME}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tt_metal
        ${CMAKE_SOURCE_DIR}/tt_metal/common
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal)
    target_compile_options(${TEST} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND METAL_TEST_TARGETS ${TEST})
endforeach()

add_subdirectory(unit_tests_common)
add_subdirectory(unit_tests)
add_subdirectory(unit_tests_fast_dispatch)
add_subdirectory(unit_tests_fast_dispatch_single_chip_multi_queue)
add_subdirectory(unit_tests_frequent)

add_custom_target(metal_tests DEPENDS ${METAL_TEST_TARGETS} unit_tests unit_tests_fast_dispatch unit_tests_fast_dispatch_single_chip_multi_queue unit_tests_frequent)

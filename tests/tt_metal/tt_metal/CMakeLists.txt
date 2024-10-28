add_library(test_metal_common_libs INTERFACE)
target_link_libraries(
    test_metal_common_libs
    INTERFACE
        tt_metal
        test_common_libs
)

set(TT_METAL_TESTS_SRCS
    test_bmm.cpp
    test_add_two_ints.cpp
    # test_compile_args.cpp         <- not tested in run_tt_metal.py
    test_eltwise_binary.cpp
    # test_eltwise_unary.cpp        <- not tested in run_tt_metal.py
    test_matmul_single_tile_bfp8b.cpp
    test_matmul_single_tile_output_in_l1.cpp
    test_dram_loopback_single_core.cpp
    test_datacopy_bfp8b.cpp
    test_datacopy.cpp
    test_datacopy_output_in_l1.cpp
    test_dataflow_cb.cpp
    test_transpose_hc.cpp
    test_multiple_programs.cpp
    test_multi_core_kernel.cpp
    test_interleaved_layouts.cpp
    test_interleaved_l1_buffer.cpp
    test_bcast.cpp
    test_generic_binary_reader_matmul_large_block.cpp
    # test_3x3conv_as_matmul_large_block.cpp    <- not tested in run_tt_metal.py
    # test_l1_to_l1_multi_core.cpp                <- test borked
    test_dram_copy_sticks_multi_core.cpp
    test_untilize_eltwise_binary.cpp
    test_bfp8_conversion.cpp
    # test_bfp4_conversion.cpp        <- not tested in run_tt_metal.py
    test_core_range_set.cpp
    test_compile_sets_kernel_binaries.cpp
    test_compile_program.cpp
    test_kernel_path_env_var.cpp
    test_clean_init.cpp
    test_create_kernel_from_string.cpp
)

foreach(TEST_SRC ${TT_METAL_TESTS_SRCS})
    get_filename_component(TEST ${TEST_SRC} NAME_WE)
    add_executable(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
    target_link_libraries(${TEST} PUBLIC test_metal_common_libs)
    target_include_directories(
        ${TEST}
        PRIVATE
            ${UMD_HOME}
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tt_metal/common
            ${PROJECT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        ${TEST}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
    list(APPEND METAL_TEST_TARGETS ${TEST})
endforeach()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_fast_dispatch)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_fast_dispatch_single_chip_multi_queue)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_frequent)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf_microbenchmark)

add_custom_target(
    metal_tests
    DEPENDS
        ${METAL_TEST_TARGETS}
        unit_tests
        unit_tests_fast_dispatch
        unit_tests_fast_dispatch_single_chip_multi_queue
        unit_tests_frequent
        metal_perf_microbenchmark_tests
        unit_tests_galaxy
)

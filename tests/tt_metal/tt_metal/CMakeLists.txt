add_library(test_metal_common_libs INTERFACE)
target_link_libraries(
    test_metal_common_libs
    INTERFACE
        Metalium::Metal
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
    test_stress_noc_mcast.cpp
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
    test_clean_init.cpp
)

foreach(TEST_SRC ${TT_METAL_TESTS_SRCS})
    get_filename_component(TEST ${TEST_SRC} NAME_WE)
    add_executable(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
    target_link_libraries(${TEST} PUBLIC test_metal_common_libs)
    target_include_directories(
        ${TEST}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data_movement)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debug_tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/eth)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/integration)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf_microbenchmark)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/noc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lightmetal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sfpi)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/misc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tunneling)

add_custom_target(
    metal_tests
    DEPENDS
        ${METAL_TEST_TARGETS}
        metal_perf_microbenchmark_tests
        unit_tests_api
        unit_tests_data_movement
        unit_tests_debug_tools
        unit_tests_device
        unit_tests_dispatch
        unit_tests_eth
        unit_tests_integration
        unit_tests_llk
        unit_tests_noc
        unit_tests_lightmetal
        unit_tests_sfpi
        unit_tests_misc
        unit_tests_tunneling
)

add_library(metalium_test_files INTERFACE)
# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    metalium_test_files
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels test_kernels/*)
target_sources(
    metalium_test_files
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../.. # FIXME
        FILES ${kernels}
)

install(
    TARGETS
        metalium_test_files
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium # FIXME: fix the include paths for jit_build
        COMPONENT metalium-validation
)

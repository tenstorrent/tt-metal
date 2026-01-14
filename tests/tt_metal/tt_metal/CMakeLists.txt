add_library(test_metal_common_libs INTERFACE)
target_link_libraries(
    test_metal_common_libs
    INTERFACE
        Metalium::Metal
        test_common_libs
)

# Consolidated legacy gtest executable
set(UNIT_TESTS_LEGACY_SRC
    test_add_two_ints.cpp
    test_bcast.cpp
    test_bfp8_conversion.cpp
    test_bmm.cpp
    test_clean_init.cpp
    test_compile_program.cpp
    test_compile_sets_kernel_binaries.cpp
    test_core_range_set.cpp
    test_datacopy.cpp
    test_datacopy_bfp8b.cpp
    test_datacopy_output_in_l1.cpp
    test_dataflow_cb.cpp
    test_dram_copy_sticks_multi_core.cpp
    test_dram_loopback_single_core.cpp
    test_eltwise_binary.cpp
    test_generic_binary_reader_matmul_large_block.cpp
    test_interleaved_l1_buffer.cpp
    test_interleaved_layouts.cpp
    test_matmul_single_tile_bfp8b.cpp
    test_matmul_single_tile_output_in_l1.cpp
    test_multi_core_kernel.cpp
    test_multiple_programs.cpp
    test_sdpa_reduce_c.cpp
    test_single_dm_l1_write.cpp
    test_stress_noc_mcast.cpp
    test_transpose_hc.cpp
    test_untilize_eltwise_binary.cpp
    test_unaligned_read_write_core.cpp
)

add_executable(unit_tests_legacy)
target_sources(unit_tests_legacy PRIVATE ${UNIT_TESTS_LEGACY_SRC})
target_link_libraries(unit_tests_legacy PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_legacy
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_legacy
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
list(APPEND METAL_TEST_TARGETS unit_tests_legacy)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data_movement)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debug_tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/eth)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hal_codegen)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/integration)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf_microbenchmark)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/noc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lightmetal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sfpi)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/misc)

add_custom_target(
    metal_tests
    DEPENDS
        ${METAL_TEST_TARGETS}
        metal_perf_microbenchmark_tests
        unit_tests_api
        unit_tests_data_movement
        unit_tests_debug_tools
        unit_tests_inspector
        unit_tests_device
        unit_tests_dispatch
        unit_tests_eth
        unit_tests_integration
        unit_tests_llk
        unit_tests_noc
        unit_tests_lightmetal
        unit_tests_sfpi
        unit_tests_misc
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
file(
    GLOB_RECURSE kernels
    test_kernels/*
    perf_microbenchmark/routing/kernels/*
)
target_sources(
    metalium_test_files
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../.. # FIXME
        FILES
            ${kernels}
            perf_microbenchmark/common/kernel_utils.hpp
            perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp
            perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp
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

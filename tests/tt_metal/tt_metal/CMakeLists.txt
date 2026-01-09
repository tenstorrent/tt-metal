add_library(test_metal_common_libs INTERFACE)
target_link_libraries(
    test_metal_common_libs
    INTERFACE
        Metalium::Metal
        test_common_libs
)

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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_legacy)

add_custom_target(
    metal_tests
    DEPENDS
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
        unit_tests_legacy
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

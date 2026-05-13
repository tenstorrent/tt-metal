# Standalone llrt smoke tests (non-gtest binaries; require real silicon + external HW).

add_executable(test_external_cmac_smoke test_external_cmac_smoke.cpp)

target_precompile_headers(test_external_cmac_smoke REUSE_FROM metal_common_pch)

target_link_libraries(test_external_cmac_smoke PRIVATE Metalium::Metal)

target_include_directories(
    test_external_cmac_smoke
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    test_external_cmac_smoke
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/llrt
)

add_executable(test_external_cmac_soak test_external_cmac_soak.cpp)
target_precompile_headers(test_external_cmac_soak REUSE_FROM metal_common_pch)
target_link_libraries(test_external_cmac_soak PRIVATE Metalium::Metal)
target_include_directories(
    test_external_cmac_soak
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)
set_target_properties(
    test_external_cmac_soak
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/llrt
)

add_executable(test_external_cmac_rx_idle test_external_cmac_rx_idle.cpp)
target_precompile_headers(test_external_cmac_rx_idle REUSE_FROM metal_common_pch)
target_link_libraries(test_external_cmac_rx_idle PRIVATE Metalium::Metal)
target_include_directories(
    test_external_cmac_rx_idle
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)
set_target_properties(
    test_external_cmac_rx_idle
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/llrt
)

add_executable(test_external_cmac_dma_pull_probe test_external_cmac_dma_pull_probe.cpp)
target_precompile_headers(test_external_cmac_dma_pull_probe REUSE_FROM metal_common_pch)
target_link_libraries(test_external_cmac_dma_pull_probe PRIVATE Metalium::Metal)
target_include_directories(
    test_external_cmac_dma_pull_probe
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)
set_target_properties(
    test_external_cmac_dma_pull_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/llrt
)

# Mellanox-side smoke test (DPDK). Built only if libdpdk is detectable via
# pkg-config so non-Mellanox hosts still configure cleanly.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(DPDK QUIET libdpdk)
endif()
if(DPDK_FOUND)
    add_executable(test_external_cmac_smoke_mlx test_external_cmac_smoke_mlx.cpp)
    target_include_directories(test_external_cmac_smoke_mlx SYSTEM PRIVATE ${DPDK_INCLUDE_DIRS})
    target_compile_options(test_external_cmac_smoke_mlx PRIVATE ${DPDK_CFLAGS_OTHER})
    target_link_libraries(test_external_cmac_smoke_mlx PRIVATE ${DPDK_LIBRARIES})
    target_link_directories(test_external_cmac_smoke_mlx PRIVATE ${DPDK_LIBRARY_DIRS})
    set_target_properties(
        test_external_cmac_smoke_mlx
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/llrt
    )
    message(STATUS "test_external_cmac_smoke_mlx: libdpdk ${DPDK_VERSION} found, target enabled")
else()
    message(STATUS "test_external_cmac_smoke_mlx: libdpdk not found, skipping target")
endif()

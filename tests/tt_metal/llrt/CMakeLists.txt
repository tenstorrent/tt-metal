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

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_jit_build ${UNIT_TESTS_JIT_BUILD_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_jit_build)
target_precompile_headers(unit_tests_jit_build REUSE_FROM metal_common_pch)

target_link_libraries(unit_tests_jit_build PUBLIC test_metal_common_libs)

set_target_properties(
    unit_tests_jit_build
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

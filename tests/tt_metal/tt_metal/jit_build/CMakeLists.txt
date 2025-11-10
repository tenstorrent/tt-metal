set(UNIT_TESTS_JIT_BUILD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_depend.cpp)

add_executable(unit_tests_jit_build ${UNIT_TESTS_JIT_BUILD_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_jit_build)

target_link_libraries(unit_tests_jit_build PUBLIC test_metal_common_libs)

set_target_properties(
    unit_tests_jit_build
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_jit_build ${UNIT_TESTS_JIT_BUILD_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_jit_build)
target_precompile_headers(unit_tests_jit_build REUSE_FROM metal_common_pch)

target_link_libraries(unit_tests_jit_build PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_jit_build
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(
    unit_tests_jit_build
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

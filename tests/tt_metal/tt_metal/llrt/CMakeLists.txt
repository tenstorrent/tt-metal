# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_llrt)
TT_ENABLE_UNITY_BUILD(unit_tests_llrt)
target_precompile_headers(unit_tests_llrt REUSE_FROM metal_common_pch)
target_sources(unit_tests_llrt PRIVATE ${UNIT_TESTS_LLRT_SOURCES})
target_include_directories(
    unit_tests_llrt
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(unit_tests_llrt PRIVATE test_metal_common_libs)

set_target_properties(
    unit_tests_llrt
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_noc_debugging ${UNIT_TESTS_NOC_DEBUGGING_SRC})

TT_ENABLE_UNITY_BUILD(unit_tests_noc_debugging)
target_precompile_headers(unit_tests_noc_debugging REUSE_FROM metal_common_pch)

target_link_libraries(unit_tests_noc_debugging PRIVATE test_metal_common_libs)

target_include_directories(
    unit_tests_noc_debugging
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    unit_tests_noc_debugging
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

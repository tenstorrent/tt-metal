add_executable(unit_tests_didt_max_util test_max_util_workload.cpp)
tt_reuse_precompile_headers(unit_tests_didt_max_util metal_test_pch)

target_link_libraries(
    unit_tests_didt_max_util
    PUBLIC
        Metalium::Metal
        test_common_libs
)

target_include_directories(
    unit_tests_didt_max_util
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    unit_tests_didt_max_util
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/didt
)

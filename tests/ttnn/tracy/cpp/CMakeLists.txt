add_executable(test_get_programs_perf_data test_get_programs_perf_data.cpp)

target_link_libraries(
    test_get_programs_perf_data
    PUBLIC
        ttnn
        test_common_libs
        Python3::Python
)

set_target_properties(
    test_get_programs_perf_data
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/tracy/
)

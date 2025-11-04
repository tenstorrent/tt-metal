add_executable(test_get_ops_perf_data test_get_ops_perf_data.cpp)

target_include_directories(
    test_get_ops_perf_data
    BEFORE
    PRIVATE
    # "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    # ${PROJECT_SOURCE_DIR}/tests
    # ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
    # ${PROJECT_SOURCE_DIR}/ttnn/cpp
    # ${PROJECT_SOURCE_DIR}/ttnn/
    # ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    test_get_ops_perf_data
    PUBLIC
        ttnn
        test_common_libs
        Python3::Python
)

set_target_properties(
    test_get_ops_perf_data
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/tracy/
)

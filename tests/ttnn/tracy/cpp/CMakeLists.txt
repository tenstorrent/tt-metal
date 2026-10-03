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
tt_reuse_precompile_headers(test_get_programs_perf_data TTNN::PCHLean)

add_executable(
    test_streaming_profiler_sync_workloads
    streaming_profiler_sync/test_streaming_profiler_sync_workloads.cpp
)
target_link_libraries(
    test_streaming_profiler_sync_workloads
    PRIVATE
        TTNN::CPP
        tt_metal
)
target_include_directories(
    test_streaming_profiler_sync_workloads
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${CMAKE_CURRENT_SOURCE_DIR}/streaming_profiler_sync
)
set_target_properties(
    test_streaming_profiler_sync_workloads
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/tracy/
)

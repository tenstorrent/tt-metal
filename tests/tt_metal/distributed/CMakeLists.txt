add_executable(distributed_unit_tests)
target_sources(
    distributed_unit_tests
    PRIVATE
        test_end_to_end_eltwise.cpp
        test_distributed_host_buffer.cpp
        test_mesh_buffer.cpp
        test_mesh_coord.cpp
        test_mesh_device.cpp
        test_mesh_device_reshape.cpp
        test_mesh_device_view.cpp
        test_mesh_workload.cpp
        test_mesh_socket.cpp
        test_mesh_sub_device.cpp
        test_mesh_allocator.cpp
        test_mesh_events.cpp
        test_mesh_trace.cpp
        test_thread_pool.cpp
        utils.cpp
)

target_link_libraries(
    distributed_unit_tests
    PRIVATE
        tt_metal
        test_common_libs
)

target_include_directories(
    distributed_unit_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    distributed_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

add_executable(distributed_benchmarks)
target_sources(
    distributed_benchmarks
    PRIVATE
        benchmark_thread_pool.cpp
        benchmark_distributed_host_buffer.cpp
)

target_link_libraries(
    distributed_benchmarks
    PRIVATE
        tt_metal
        test_common_libs
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(
    distributed_benchmarks
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    distributed_benchmarks
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

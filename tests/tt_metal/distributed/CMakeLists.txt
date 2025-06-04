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

add_executable(thread_pool_benchmark)
target_sources(thread_pool_benchmark PRIVATE benchmark_thread_pool.cpp)

target_link_libraries(
    thread_pool_benchmark
    PRIVATE
        tt_metal
        test_common_libs
        benchmark::benchmark
)

target_include_directories(
    thread_pool_benchmark
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    thread_pool_benchmark
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

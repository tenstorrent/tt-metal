# Shared layer-completion impl source list (LAYER_COMPLETION_IMPL_SRCS / LAYER_COMPLETION_DIR).
include(${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_d_p/tt/runners/pipelined_prefill/sources.cmake)

set(DISTRIBUTED_UNIT_TEST_SOURCES
    test_auto_mpi_init.cpp
    test_control_plane_local_mesh_binding.cpp
    test_distributed_coordinate_translator.cpp
    test_end_to_end_eltwise.cpp
    test_distributed_host_buffer.cpp
    test_maybe_remote.cpp
    test_mesh_buffer.cpp
    test_mesh_coord.cpp
    test_mesh_device.cpp
    test_mesh_device_reshape.cpp
    test_mesh_device_view.cpp
    test_mesh_workload.cpp
    test_mesh_socket.cpp
    test_hd_sockets.cpp
    test_layer_completion_layout.cpp
    test_layer_completion_queue.cpp
    test_layer_completion_queue_mpsc.cpp
    test_layer_completion_reorder_buffer.cpp
    test_layer_completion_router.cpp
    test_mesh_sub_device.cpp
    test_mesh_allocator.cpp
    test_mesh_events.cpp
    test_mesh_trace.cpp
    test_thread_pool.cpp
    test_dispatch_context.cpp
    utils.cpp
    # The layer-completion impls live with the pipelined-prefill feature (models/demos/...); compile
    # them in so the queue/reorder/router gtests link without pulling in the feature's standalone .so.
    ${LAYER_COMPLETION_IMPL_SRCS}
)
# MPI sub-context tests require the MPI distributed build (matches tt_metal/distributed USE_MPI).
if(ENABLE_DISTRIBUTED)
    list(APPEND DISTRIBUTED_UNIT_TEST_SOURCES test_mpi_subcontext.cpp)
endif()

add_executable(distributed_unit_tests)
target_sources(distributed_unit_tests PRIVATE ${DISTRIBUTED_UNIT_TEST_SOURCES})

target_link_libraries(
    distributed_unit_tests
    PRIVATE
        tt_metal
        test_common_libs
)
if(ENABLE_DISTRIBUTED)
    target_link_libraries(distributed_unit_tests PRIVATE OpenMPI::MPI)
endif()

target_include_directories(
    distributed_unit_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal/distributed # named_shm.hpp (used by the layer_completion queue impl)
        ${LAYER_COMPLETION_DIR} # layer-completion feature headers (queue/reorder/router/ring_layout/message)
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
        benchmark_hd_sockets.cpp
        hd_socket_test_utils.cpp
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
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    distributed_benchmarks
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

add_executable(multiproc_loopback_test)
target_sources(multiproc_loopback_test PRIVATE test_multiproc_loopback.cpp)

target_link_libraries(
    multiproc_loopback_test
    PRIVATE
        tt_metal
        test_common_libs
)

target_include_directories(
    multiproc_loopback_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    multiproc_loopback_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

add_subdirectory(multiprocess)

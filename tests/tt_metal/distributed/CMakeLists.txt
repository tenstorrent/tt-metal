# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

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

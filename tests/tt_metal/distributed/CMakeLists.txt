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

# One target per leg benchmark. Each keeps its own main so DistributedContext::create sees
# argv, which is why none of them can be a source in distributed_benchmarks.
# name | source | needs MPI | needs hd_socket_test_utils.cpp
set(LEG_BENCHMARKS
    "d2h_leg_benchmark|benchmark_d2h_leg.cpp|OFF|ON"
    "h2d_leg_benchmark|benchmark_h2d_leg.cpp|OFF|ON"
    "h2h_leg_benchmark|benchmark_h2h_leg.cpp|ON|OFF"
    "d2h2h2d_leg_benchmark|benchmark_d2h2h2d_leg.cpp|ON|ON"
)

foreach(spec IN LISTS LEG_BENCHMARKS)
    string(REPLACE "|" ";" parts "${spec}")
    list(GET parts 0 name)
    list(GET parts 1 src)
    list(GET parts 2 needs_mpi)
    list(GET parts 3 needs_utils)

    # The MPI legs are two ranks over an RdmaWindow, so a non-MPI build skips them.
    if(needs_mpi AND NOT TARGET OpenMPI::MPI)
        continue()
    endif()

    add_executable(${name})
    target_sources(${name} PRIVATE ${src})
    # Not a library: its .cpp is otherwise compiled only into distributed_benchmarks.
    if(needs_utils)
        target_sources(${name} PRIVATE hd_socket_test_utils.cpp)
    endif()

    target_link_libraries(
        ${name}
        PRIVATE
            tt_metal
            test_common_libs
            benchmark::benchmark
    )
    if(needs_mpi)
        target_link_libraries(${name} PRIVATE OpenMPI::MPI)
    endif()

    # TT_DIRECT_KERNEL_DIR: the kernels stay with the d2h2h2d socket tests, and CreateKernel
    # takes an absolute path, so they are found without a copy step.
    target_compile_definitions(
        ${name}
        PRIVATE
            TT_METAL_HOST_BRIDGE=1
            TT_DIRECT_KERNEL_DIR="${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost/d2h2h2d_socket"
    )

    target_include_directories(
        ${name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}
    )

    set_target_properties(
        ${name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
    )
endforeach()

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

set(UNIT_TESTS_DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_coord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_device_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_sub_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp # Common utilities
)

set(THREAD_POOL_BENCHMARK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_thread_pool.cpp)

# Function to create the main distributed unit test binary
function(create_unit_test_executable arch_name)
    set(exec_name distributed_unit_tests_${arch_name})
    string(REPLACE "wormhole" "wormhole_b0" exec_name "${exec_name}")

    add_executable(${exec_name} ${UNIT_TESTS_DISTRIBUTED_SRC})

    target_link_libraries(
        ${exec_name}
        PRIVATE
            tt_metal
            test_common_libs
    )

    target_include_directories(
        ${exec_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${arch_name}
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
    )

    set_target_properties(
        ${exec_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
    )
endfunction()

# Function to create a separate binary for thread_pool benchmarking
function(create_thread_pool_benchmark_executable)
    set(exec_name thread_pool_benchmark)
    string(REPLACE "wormhole" "wormhole_b0" exec_name "${exec_name}")

    add_executable(${exec_name} ${THREAD_POOL_BENCHMARK_SRC})

    target_link_libraries(
        ${exec_name}
        PRIVATE
            tt_metal
            test_common_libs
            benchmark::benchmark
    )

    target_include_directories(
        ${exec_name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
    )

    set_target_properties(
        ${exec_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
    )
endfunction()

# Define the architectures for which to create test executables
set(ARCHITECTURES
    "grayskull"
    "wormhole"
    "blackhole"
)

# Create distributed test executables for each architecture
foreach(arch IN LISTS ARCHITECTURES)
    create_unit_test_executable(${arch})
endforeach()

# Create thread pool benchmark executable
create_thread_pool_benchmark_executable()

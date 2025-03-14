set(UNIT_TESTS_API_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/test_free_list_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/test_free_list_opt_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/test_l1_banking_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer/test_CircularBuffer_allocation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer/test_CircularBuffer_creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer/test_CircularBuffer_non_blocking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_adjacent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_contains.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_intersects.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRange_merge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_construct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_contains.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_intersects.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord/test_CoreRangeSet_merge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution_spec/test_distribution_spec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_banked.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bit_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_buffer_region.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compile_time_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_direct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dram_to_l1_multicast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_global_circular_buffers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_global_semaphores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel_creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_noc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_semaphores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shape_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sharded_l1_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simple_dram_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simple_l1_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tilize_untilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_worker_config_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_blockfloat_common.cpp
)

# Define the function to create test executables for each architecture
function(create_unit_test_executable arch_name)
    # Define the test executable name using the architecture name
    set(exec_name unit_tests_api_${arch_name})
    string(REPLACE "wormhole" "wormhole_b0" exec_name "${exec_name}")

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_API_SRC})

    # Enable unity build for the executable
    TT_ENABLE_UNITY_BUILD(${exec_name})

    # Link libraries
    target_link_libraries(
        ${exec_name}
        PRIVATE
            test_metal_common_libs
            Boost::smart_ptr
    )

    # Set include directories
    target_include_directories(
        ${exec_name}
        BEFORE
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${arch_name}
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/common
    )

    # Set runtime output directory
    set_target_properties(
        ${exec_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
endfunction()

# Define the architectures for which to create test executables
set(ARCHITECTURES
    "wormhole"
    "blackhole"
)

# Create a test executable for each architecture
foreach(arch IN LISTS ARCHITECTURES)
    create_unit_test_executable(${arch})
endforeach()

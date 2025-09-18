# Create object library first
add_library(unit_tests_api_lib OBJECT)
add_library(TT::Metalium::Test::API ALIAS unit_tests_api_lib)

# Use target_sources to gather the necessary sources
target_sources(
    unit_tests_api_lib
    PRIVATE
        allocator/test_free_list_opt_allocator.cpp
        allocator/test_l1_banking_allocator.cpp
        allocator/test_overlapped_bank_manager.cpp
        circular_buffer/test_CircularBuffer_allocation.cpp
        circular_buffer/test_CircularBuffer_creation.cpp
        circular_buffer/test_CircularBuffer_non_blocking.cpp
        circular_buffer/test_CircularBuffer_wrapping.cpp
        core_coord/test_CoreRange_adjacent.cpp
        core_coord/test_CoreRange_contains.cpp
        core_coord/test_CoreRange_intersects.cpp
        core_coord/test_CoreRange_iterator.cpp
        core_coord/test_CoreRange_merge.cpp
        core_coord/test_CoreRangeSet_construct.cpp
        core_coord/test_CoreRangeSet_contains.cpp
        core_coord/test_CoreRangeSet_intersects.cpp
        core_coord/test_CoreRangeSet_merge.cpp
        distribution_spec/test_buffer_distribution_spec.cpp
        test_banked.cpp
        test_bit_utils.cpp
        test_buffer_region.cpp
        test_compile_time_args.cpp
        test_direct.cpp
        test_dram_to_l1_multicast.cpp
        test_dram.cpp
        test_global_circular_buffers.cpp
        test_global_semaphores.cpp
        test_host_buffer.cpp
        test_kernel_compile_cache.cpp
        test_kernel_creation.cpp
        test_memory_pin.cpp
        test_noc.cpp
        test_runtime_args.cpp
        test_semaphores.cpp
        test_shape_base.cpp
        test_sharded_l1_buffer.cpp
        test_simple_dram_buffer.cpp
        test_simple_l1_buffer.cpp
        test_soc_descriptor.cpp
        test_tilize_untilize.cpp
        test_worker_config_buffer.cpp
        test_blockfloat_common.cpp
        test_duplicate_kernel.cpp
)

target_include_directories(
    unit_tests_api_lib
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Link libraries to the object library
target_link_libraries(
    unit_tests_api_lib
    PRIVATE
        test_metal_common_libs
        Boost::smart_ptr
)

TT_ENABLE_UNITY_BUILD(unit_tests_api_lib)

# Create the test executable
add_executable(unit_tests_api)

# Link libraries to the executable
target_link_libraries(unit_tests_api PRIVATE unit_tests_api_lib)

# Set runtime output directory
set_target_properties(
    unit_tests_api
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

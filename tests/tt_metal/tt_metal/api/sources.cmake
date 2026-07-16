# Source files for tt_metal api tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_API_SOURCES
    allocator/test_free_list_opt_allocator.cpp
    allocator/test_l1_banking_allocator.cpp
    allocator/test_overlapped_bank_manager.cpp
    allocator/test_per_core_bank_manager.cpp
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
    dataflow_buffer/test_alias_dataflow_buffer.cpp
    dataflow_buffer/test_dataflow_buffer_apis.cpp
    dataflow_buffer/test_dataflow_buffer_base.cpp
    dataflow_buffer/test_dataflow_buffer_multinode.cpp
    dataflow_buffer/test_dataflow_buffer_intra.cpp
    dataflow_buffer/test_dataflow_buffer_edge_cases.cpp
    dataflow_buffer/test_dataflow_buffer_overrides.cpp
    dataflow_buffer/test_dataflow_buffer_configs.cpp
    dataflow_buffer/test_borrowed_memory_dataflow_buffer.cpp
    distribution_spec/test_buffer_distribution_spec.cpp
    metal2_host_api/test_program_spec.cpp
    metal2_host_api/test_program_spec_hw.cpp
    metal2_host_api/test_scratchpad_hw.cpp
    metal2_host_api/test_program_run_args.cpp
    metal2_host_api/test_table.cpp
    test_kernel_thread_sync.cpp
    test_banked.cpp
    test_bit_utils.cpp
    test_filesystem_utils.cpp
    test_graph_tracking.cpp
    test_buffer_region.cpp
    test_compile_time_args.cpp
    test_compile_defines.cpp
    test_compiler_include_paths.cpp
    test_direct.cpp
    test_dram_kernels.cpp
    test_dram_sender_global_cb.cpp
    test_dram_subchannel_helper.cpp
    test_dram_to_l1_multicast.cpp
    test_dram.cpp
    test_global_circular_buffers.cpp
    test_global_semaphores.cpp
    test_host_buffer.cpp
    test_kernel_compile_cache.cpp
    test_kernel_creation.cpp
    test_offline_kernel_compile.cpp
    test_memory_pin.cpp
    test_noc.cpp
    test_runtime_args.cpp
    test_named_runtime_args.cpp
    test_semaphores.cpp
    test_shape_base.cpp
    test_shape.cpp
    test_sharded_l1_buffer.cpp
    test_simple_dram_buffer.cpp
    test_tensor_accessor_default_page_size.cpp
    test_simple_l1_buffer.cpp
    test_soc_descriptor.cpp
    test_stream_scratch_register.cpp
    test_tilize_untilize.cpp
    test_worker_config_buffer.cpp
    test_blockfloat_common.cpp
    test_descriptor_patching.cpp
    test_duplicate_kernel.cpp
    test_core_local_mem_api.cpp
    test_zero_memory_api.cpp
    disaggregation/test_kv_chunk_address_table.cpp
)

# tt-emule ASAN sanitizer tests. These EXPECT_DEATH tests assert on the emule
# ASAN panic (e.g. "Illegal Semaphore Access") and JIT kernels that reference
# emule-only defines/intrinsics (EMULE_SEM_BASE, __emule_local_l1_to_ptr). They
# only build/pass under the emule backend; on ttsim/HW the kernels fail to
# compile and the death tests fail. Gate them so they never enter the non-emule
# unit_tests_api binary.
if(TT_METAL_USE_EMULE)
    list(
        APPEND
        UNIT_TESTS_API_SOURCES
        emule/test_alignment_writes.cpp
        emule/test_cb_leak.cpp
        emule/test_cb_pages.cpp
        emule/test_host_alignment.cpp
        emule/test_metadata_size.cpp
        emule/test_noc_without_barrier.cpp
        emule/test_padded_write.cpp
        emule/test_semaphore_write.cpp
        emule/test_tensor_bad_access.cpp
        emule/test_valid_mem_wrong_alloc.cpp
        emule/test_write_beyond_res_pages.cpp
        emule/test_write_outside_tensor.cpp
    )
endif()

# Runtime tensor tests build into their own executable (unit_tests_tensor),
# mirroring unit_tests_ttnn_tensor, so they stay out of the tt-metalium smoke binary.
set(UNIT_TESTS_API_TENSOR_SOURCES
    tensor/common_tensor_test_utils.cpp
    tensor/test_tensor_sharding.cpp
    tensor/test_host_tensor.cpp
    tensor/test_host_tensor_to_layout.cpp
    tensor/test_mesh_tensor.cpp
    tensor/test_tensor_types.cpp
    tensor/test_tensor_layout.cpp
    tensor/test_create_tensor.cpp
    tensor/test_create_tensor_with_layout.cpp
    tensor/test_tensor_nd_sharding.cpp
    tensor/test_vector_conversion.cpp
)

# Program Factory Refactoring Tracker

Total program factories found: 141
Total files: 91
Files with multiple factories: 23

## Files with Multiple Program Factories

These files contain multiple program factory functions and may need special attention:

### ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp (3 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| embeddings_fused | 77 | TODO |
| embeddings_rm | 375 | TODO |
| embeddings_tilized_indices | 635 | TODO |

### ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op_multi_core.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| update_cache_multi_core | 19 | TODO |
| fill_cache_multi_core | 319 | TODO |

### ttnn/cpp/ttnn/operations/reduction/topk/device/topk_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| topk_single_core_interleaved | 17 | TODO |
| topk_multicore_interleaved | 264 | TODO |

### ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| argmax_single_core | 89 | TODO |
| argmax_multi_core | 264 | TODO |

### ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| rotary_embedding_llama_multi_core | 17 | TODO |
| rotary_embedding_llama_multi_core_sharded | 335 | TODO |

### ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| multi_core_split_query_key_value_and_split_heads | 16 | TODO |
| multi_core_split_query_key_value_and_split_heads_sharded | 211 | TODO |

### ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| multi_core_nlp_concat_heads_decode | 16 | TODO |
| multi_core_nlp_concat_heads_decode_subcoregrids | 153 | TODO |

### ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp (3 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| multi_core_nlp_create_qkv_heads_decode_interleaved_input | 58 | TODO |
| multi_core_nlp_create_qkv_heads_decode_sharded_input | 215 | TODO |
| multi_core_nlp_create_qkv_heads_decode_sharded_input_subcoregrid | 506 | TODO |

### ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| paged_tiled_fused_update_cache_multi_core | 69 | TODO |
| paged_row_major_fused_update_cache_multi_core | 533 | TODO |

### ttnn/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp (3 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| slice_write_rm_sharded_input_multi_core | 354 | TODO |
| slice_write_tiled_sharded_input_multi_core | 682 | TODO |
| slice_write_rm_interleaved_multi_core | 839 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp (4 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| slice_rm_multi_core | 190 | TODO |
| slice_rm_strided_single_core_n_dims | 326 | TODO |
| slice_rm_multi_core_sharded | 598 | TODO |
| slice_tile_multi_core | 873 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp (6 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| untilize_multi_core_sub_core_grids | 35 | TODO |
| untilize_multi_core_parallelize_column | 220 | TODO |
| untilize_multi_core_block | 453 | TODO |
| untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical | 740 | TODO |
| untilize_multi_core | 867 | TODO |
| untilize_single_core | 1313 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp (5 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| untilize_with_unpadding_single_core | 24 | TODO |
| untilize_with_unpadding_multi_core_block_interleaved | 217 | TODO |
| untilize_with_unpadding_multi_core_col_interleaved | 505 | TODO |
| untilize_with_unpadding_multi_core_interleaved | 662 | TODO |
| untilize_with_unpadding_multi_core_sharded | 882 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp (4 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| tilize_single_core | 21 | TODO |
| tilize_multi_core_block | 162 | TODO |
| tilize_multi_core_interleaved | 437 | TODO |
| tilize_multi_core_sharded | 628 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/repeat/device/host/repeat_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| rm_repeater_last_dim | 26 | TODO |
| rm_repeater | 141 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp (5 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| s2s_tiled_concat_two_tensors_height_multi_core | 35 | TODO |
| s2s_rm_concat_two_tensors_height_multi_core | 248 | TODO |
| s2s_concat_multi_core | 445 | TODO |
| s2i_rm_concat_multi_core | 566 | TODO |
| concat_multi_core | 726 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| reshape_tile_single_core | 16 | TODO |
| reshape_rm_multi_core | 211 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp (5 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| transpose_hc_multi_core | 624 | TODO |
| transpose_hc_multi_core_sharded | 1148 | TODO |
| transpose_wh_multi_core | 1511 | TODO |
| transpose_wh_multi_core_sharded | 1755 | TODO |
| transpose_wh_multi_core_sharded_rm | 1964 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| move_multi_core_with_overlap | 67 | TODO |
| move_multi_core_sharded | 210 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp (6 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| pad_rm_reader_writer | 23 | TODO |
| pad_tile | 179 | TODO |
| pad_rm_reader_writer_multi_core | 449 | TODO |
| pad_rm_reader_writer_multi_core_v2 | 797 | TODO |
| pad_rm_sharded_height_only | 1183 | TODO |
| pad_rm_sharded_width_only | 1361 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp (4 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| tilize_with_val_padding_single_core | 55 | TODO |
| tilize_with_val_padding_multi_core_block_interleaved | 237 | TODO |
| tilize_with_val_padding_multi_core_interleaved | 517 | TODO |
| tilize_with_val_padding_multi_core_sharded | 719 | TODO |

### ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp (3 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| reshard_multi_core_same_width | 296 | TODO |
| reshard_multi_core_generic | 455 | TODO |
| reshard_multi_core_same_height | 562 | TODO |

### ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp (2 factories)

| Factory Name | Line | Status |
|--------------|------|--------|
| scale_mask_softmax_multi_core | 34 | TODO |
| scale_mask_softmax_sharded_multi_core | 591 | TODO |

## All Program Factories List

| Factory Name | File | Line | Status |
|--------------|------|------|--------|
| embeddings_fused | ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp | 77 | TODO |
| embeddings_rm | ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp | 375 | TODO |
| embeddings_tilized_indices | ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp | 635 | TODO |
| dram_prefetcher_multi_core | ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_op_multi_core.cpp | 36 | TODO |
| update_cache_multi_core | ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op_multi_core.cpp | 19 | TODO |
| fill_cache_multi_core | ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op_multi_core.cpp | 319 | TODO |
| prod_single_core | ttnn/cpp/ttnn/operations/reduction/prod/device/prod_all_program_factory.cpp | 15 | TODO |
| topk_single_core_interleaved | ttnn/cpp/ttnn/operations/reduction/topk/device/topk_program_factory.cpp | 17 | TODO |
| topk_multicore_interleaved | ttnn/cpp/ttnn/operations/reduction/topk/device/topk_program_factory.cpp | 264 | TODO |
| moe_single_core_interleaved | ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp | 14 | TODO |
| argmax_single_core | ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_program_factory.cpp | 89 | TODO |
| argmax_multi_core | ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_program_factory.cpp | 264 | TODO |
| sampling_multicore_interleaved | ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp | 15 | TODO |
| reduce_single_core_hw | ttnn/cpp/ttnn/operations/reduction/generic/device/single_core_hw/reduce_op_single_core_hw.cpp | 19 | TODO |
| reduce_multi_core_w | ttnn/cpp/ttnn/operations/reduction/generic/device/multi_core_w/reduce_op_multi_core_w.cpp | 21 | TODO |
| reduce_multi_core_h | ttnn/cpp/ttnn/operations/reduction/generic/device/multi_core_h/reduce_op_multi_core_h.cpp | 20 | TODO |
| joint_sdpa | ttnn/cpp/ttnn/operations/transformer/sdpa/device/joint_sdpa_program_factory.cpp | 27 | TODO |
| sdpa_multi_core | ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp | 26 | TODO |
| ring_joint_sdpa | ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp | 25 | TODO |
| sdpa_decode_multi_core | ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp | 25 | TODO |
| upsample_single_core | ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_singlecore.cpp | 25 | TODO |
| upsample_multi_core | ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore.cpp | 149 | TODO |
| bilinear_multi_core | ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.cpp | 65 | TODO |
| all_broadcast_async_multicore | ttnn/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_program.cpp | 41 | TODO |
| all_gather_async_llama_sharded | ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_program_minimal_variants.cpp | 559 | TODO |
| all_reduce_async_minimal_multi_core_with_workers | ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_minimal_variants.cpp | 49 | TODO |
| all_gather_concat_llama_sharded | ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_program.cpp | 59 | TODO |
| frmsnorm_multi_core_sharded | ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/multi_core/rms_allgather_pf.cpp | 60 | TODO |
| all_to_all_async_minimal | ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/all_to_all_program_factory.cpp | 134 | TODO |
| conv3d_factory | ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp | 14 | TODO |
| rotary_embedding_llama_multi_core | ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp | 17 | TODO |
| rotary_embedding_llama_multi_core_sharded | ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp | 335 | TODO |
| rotary_embedding_llama_fused_qk_multi_core_sharded | ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_program_factory.cpp | 17 | TODO |
| create_heads_combined_qkv_sharded | ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/device/create_qkv_heads_program_factory.cpp | 17 | TODO |
| all_reduce_create_qkv_heads_minimal_multi_core_with_workers | ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_program_factory.cpp | 11 | TODO |
| multi_core_split_query_key_value_and_split_heads | ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp | 16 | TODO |
| multi_core_split_query_key_value_and_split_heads_sharded | ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp | 211 | TODO |
| multi_core_nlp_create_qkv_heads_segformer | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_program_factory.cpp | 16 | TODO |
| multi_core_nlp_kv_cache_load_slice | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/device/nlp_kv_cache_load_slice_program_factory.cpp | 51 | TODO |
| multi_core_nlp_create_qkv_heads_falcon7b | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/device/nlp_create_qkv_heads_falcon7b_program_factory.cpp | 16 | TODO |
| multi_core_nlp_create_qkv_heads_vit | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_program_factory.cpp | 16 | TODO |
| multi_core_nlp_concat_heads_decode | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp | 16 | TODO |
| multi_core_nlp_concat_heads_decode_subcoregrids | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp | 153 | TODO |
| rotate_half_single_core | ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/single_core/rotate_half_program_factory.cpp | 18 | TODO |
| create_qkv_separate | ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/create_qkv_heads_from_separate_tensors_program_factory.cpp | 15 | TODO |
| rotary_embedding_multi_core | ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp | 20 | TODO |
| concatenate_heads_multi_core | ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/concatenate_heads_program_factory.hpp | 16 | TODO |
| multi_core_nlp_concat_heads | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp | 16 | TODO |
| multi_core_nlp_create_qkv_heads_decode_interleaved_input | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp | 58 | TODO |
| multi_core_nlp_create_qkv_heads_decode_sharded_input | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp | 215 | TODO |
| multi_core_nlp_create_qkv_heads_decode_sharded_input_subcoregrid | ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp | 506 | TODO |
| paged_tiled_fused_update_cache_multi_core | ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.cpp | 69 | TODO |
| paged_row_major_fused_update_cache_multi_core | ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.cpp | 533 | TODO |
| paged_fill_cache_multi_core | ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fill_cache_program_factory.cpp | 20 | TODO |
| paged_update_cache_multi_core | ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_update_cache_program_factory.cpp | 30 | TODO |
| slice_write_rm_sharded_input_multi_core | ttnn/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp | 354 | TODO |
| slice_write_tiled_sharded_input_multi_core | ttnn/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp | 682 | TODO |
| slice_write_rm_interleaved_multi_core | ttnn/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp | 839 | TODO |
| multi_core_attn_matmul | ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/attn_matmul_program_factory.cpp | 18 | TODO |
| multi_core_group_attn_matmul | ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/group_attn_matmul_program_factory.cpp | 18 | TODO |
| multi_core_convert_to_chw | ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/convert_to_chw_program_factory.cpp | 13 | TODO |
| multi_core_convert_to_hwc | ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.cpp | 16 | TODO |
| plusone_single_core | ttnn/cpp/ttnn/operations/experimental/plusone/device/plusone_program_factory.cpp | 17 | TODO |
| multi_core_ssm_prefix_scan | ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/prefix_scan_program_factory.cpp | 15 | TODO |
| multi_core_ssm_eltwise_mul | ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_program_factory.cpp | 17 | TODO |
| multi_core_ssm_1d_sum_reduce | ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp | 15 | TODO |
| padded_slice_tile_multi_core | ttnn/cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_tile_multi_core_program_factory.cpp | 36 | TODO |
| padded_slice_rm_multi_core | ttnn/cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_rm_multi_core_program_factory.cpp | 36 | TODO |
| create_program | ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp | 16 | TODO |
| create_program | ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_optimized_program_factory.cpp | 21 | TODO |
| matmul_multi_core | ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_program_factory.cpp | 20 | TODO |
| create_program_dram_sharded | ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp | 60 | TODO |
| embedding_backward_multi_core | ttnn/cpp/ttnn/operations/embedding_backward/device/embedding_backward_program_factory.cpp | 19 | TODO |
| bcast_multi_core_w | ttnn/cpp/ttnn/operations/data_movement/bcast/device/multi_core_w/bcast_op_multi_core_w.cpp | 18 | TODO |
| bcast_sharded_h_optimised | ttnn/cpp/ttnn/operations/data_movement/bcast/device/multi_core_h/bcast_op_sharded_h_optimised.cpp | 18 | TODO |
| bcast_multi_core_h | ttnn/cpp/ttnn/operations/data_movement/bcast/device/multi_core_h/bcast_op_multi_core_h.cpp | 19 | TODO |
| bcast_sharded_h | ttnn/cpp/ttnn/operations/data_movement/bcast/device/multi_core_h/bcast_op_sharded_h.cpp | 18 | TODO |
| bcast_multi_core_hw | ttnn/cpp/ttnn/operations/data_movement/bcast/device/multi_core_hw/bcast_op_multi_core_hw.cpp | 20 | TODO |
| reshape_tiled_program_factory | ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/host/reshape_tiled_program_factory.cpp | 288 | TODO |
| rm_reshape_preparer_single_risk | ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/host/reshape_rm_program_factory.cpp | 34 | TODO |
| slice_rm_multi_core | ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp | 190 | TODO |
| slice_rm_strided_single_core_n_dims | ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp | 326 | TODO |
| slice_rm_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp | 598 | TODO |
| slice_tile_multi_core | ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp | 873 | TODO |
| untilize_multi_core_sub_core_grids | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 35 | TODO |
| untilize_multi_core_parallelize_column | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 220 | TODO |
| untilize_multi_core_block | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 453 | TODO |
| untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 740 | TODO |
| untilize_multi_core | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 867 | TODO |
| untilize_single_core | ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp | 1313 | TODO |
| untilize_with_unpadding_single_core | ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp | 24 | TODO |
| untilize_with_unpadding_multi_core_block_interleaved | ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp | 217 | TODO |
| untilize_with_unpadding_multi_core_col_interleaved | ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp | 505 | TODO |
| untilize_with_unpadding_multi_core_interleaved | ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp | 662 | TODO |
| untilize_with_unpadding_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp | 882 | TODO |
| copy_multi_core | ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_program_factory.cpp | 22 | TODO |
| tilize_single_core | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp | 21 | TODO |
| tilize_multi_core_block | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp | 162 | TODO |
| tilize_multi_core_interleaved | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp | 437 | TODO |
| tilize_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp | 628 | TODO |
| rm_repeater_last_dim | ttnn/cpp/ttnn/operations/data_movement/repeat/device/host/repeat_program_factory.cpp | 26 | TODO |
| rm_repeater | ttnn/cpp/ttnn/operations/data_movement/repeat/device/host/repeat_program_factory.cpp | 141 | TODO |
| s2s_tiled_concat_two_tensors_height_multi_core | ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp | 35 | TODO |
| s2s_rm_concat_two_tensors_height_multi_core | ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp | 248 | TODO |
| s2s_concat_multi_core | ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp | 445 | TODO |
| s2i_rm_concat_multi_core | ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp | 566 | TODO |
| concat_multi_core | ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp | 726 | TODO |
| fill_rm_single_core | ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/fill_rm_op.cpp | 17 | TODO |
| reshape_tile_single_core | ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_program_factory.cpp | 16 | TODO |
| reshape_rm_multi_core | ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_program_factory.cpp | 211 | TODO |
| indexed_fill_multi_core | ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op_multi_core_program_factory.cpp | 19 | TODO |
| transpose_hc_multi_core | ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp | 624 | TODO |
| transpose_hc_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp | 1148 | TODO |
| transpose_wh_multi_core | ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp | 1511 | TODO |
| transpose_wh_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp | 1755 | TODO |
| transpose_wh_multi_core_sharded_rm | ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp | 1964 | TODO |
| split_last_dim_two_chunks_tiled | ttnn/cpp/ttnn/operations/data_movement/split/device/split_program_factory.cpp | 85 | TODO |
| non_zero_indices_single_core | ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.cpp | 22 | TODO |
| fill_pad_multi_core | ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.cpp | 28 | TODO |
| move_multi_core_with_overlap | ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp | 67 | TODO |
| move_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp | 210 | TODO |
| pad_rm_reader_writer | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 23 | TODO |
| pad_tile | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 179 | TODO |
| pad_rm_reader_writer_multi_core | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 449 | TODO |
| pad_rm_reader_writer_multi_core_v2 | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 797 | TODO |
| pad_rm_sharded_height_only | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 1183 | TODO |
| pad_rm_sharded_width_only | ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp | 1361 | TODO |
| tilize_with_val_padding_single_core | ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp | 55 | TODO |
| tilize_with_val_padding_multi_core_block_interleaved | ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp | 237 | TODO |
| tilize_with_val_padding_multi_core_interleaved | ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp | 517 | TODO |
| tilize_with_val_padding_multi_core_sharded | ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp | 719 | TODO |
| interleaved_to_sharded_multi_core | ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp | 24 | TODO |
| reshard_multi_core_same_width | ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp | 296 | TODO |
| reshard_multi_core_generic | ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp | 455 | TODO |
| reshard_multi_core_same_height | ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp | 562 | TODO |
| sharded_to_interleaved_multi_core | ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp | 21 | TODO |
| layernorm_post_allgather_multi_core | ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/multi_core/layernorm_post_all_gather_op_multi_core.cpp | 57 | TODO |
| layernorm_pre_allgather_multi_core | ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/multi_core/layernorm_pre_all_gather_op_multi_core.cpp | 51 | TODO |
| scale_mask_softmax_multi_core | ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp | 34 | TODO |
| scale_mask_softmax_sharded_multi_core | ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp | 591 | TODO |
| layernorm_multi_core | ttnn/cpp/ttnn/operations/normalization/layernorm/device/multi_core/layernorm_op_multi_core.cpp | 87 | TODO |

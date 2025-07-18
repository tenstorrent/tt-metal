# Program Factory Extraction Issues Summary

Generated on: 2025-07-10 22:35:58

Total files with multiple program factories: 24

## Files to Process

- `ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp` (3 factories)
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op_multi_core.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp` (2 factories)
- `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp` (3 factories)
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp` (3 factories)
- `ttnn/cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp` (4 factories)
- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp` (6 factories)
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp` (5 factories)
- `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp` (4 factories)
- `ttnn/cpp/ttnn/operations/data_movement/repeat/device/host/repeat_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp` (5 factories)
- `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp` (5 factories)
- `ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp` (2 factories)
- `ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp` (6 factories)
- `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp` (4 factories)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp` (3 factories)
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp` (2 factories)

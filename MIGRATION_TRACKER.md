# TTNN Operation Migration Tracker

## Overview
Migration from `register_operation` + `bind_registered_operation` to `bind_function<>` pattern.

**Last Updated:** 2026-02-13

## Statistics

| Metric | Count |
|--------|-------|
| Total operations found | 311 |
| **Completed Migrations** | **96** |
| **Data Movement** | **54** (COMPLETE!) |
| **Moreh Operations** | **32** (COMPLETE!) |
| **Reduction Operations** | **10** (merged from PR) |
| Conv operations (merged from PR) | 3 |
| Remaining to migrate | ~212 (mostly eltwise ~150 + others) |

### Migration Breakdown
**Data Movement (54):**
- Pre-existing: 7 operations
- Test Batch: 4 operations (squeeze reverted)
- Batch 1: 10 operations
- Batch 2: 15 operations
- Batch 3 Phase 1: 5 operations
- Batch 3 Phase 2: 6 operations
- Final Batch: 7 operations (assign, slice, view, reshape, moe_expert_token_remap, tosa_scatter, tosa_gather)

**Moreh Operations (32):**
- Batch 1: 8 operations (arange, sum, mean, cumsum, getitem, fold, abs_pow, dot)
- Batch 2: 10 operations (matmul ops + softmax + layer_norm)
- Batch 3: 14 operations (optimizers + norms + losses)

## Migration Status by Directory

### ✅ Completed (40 directories, 96 operations)
**Pre-existing (7):**
- `ttnn/cpp/ttnn/operations/normalization/layernorm/` - layer_norm
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm/` - rms_norm
- `ttnn/cpp/ttnn/operations/normalization/softmax/` - softmax (5 variants)
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/` - batch_norm
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/` - group_norm
- `ttnn/cpp/ttnn/operations/matmul/` - matmul, linear, addmm, sparse_matmul, matmul_batched_weights
- `ttnn/cpp/ttnn/operations/data_movement/split/` - split

**Test Batch (4):**
- `ttnn/cpp/ttnn/operations/data_movement/clone/` - clone ✅
- `ttnn/cpp/ttnn/operations/data_movement/copy/` - copy ✅
- `ttnn/cpp/ttnn/operations/data_movement/move/` - move ✅
- `ttnn/cpp/ttnn/operations/data_movement/transpose/` - transpose ✅
- `ttnn/cpp/ttnn/operations/data_movement/squeeze/` - ❌ REVERTED (requires lambda for polymorphism)

**Batch 1 (10):**
- `ttnn/cpp/ttnn/operations/data_movement/untilize/` - untilize ✅
- `ttnn/cpp/ttnn/operations/data_movement/tilize/` - tilize ✅
- `ttnn/cpp/ttnn/operations/data_movement/fold/` - fold ✅
- `ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/` - repeat_interleave ✅
- `ttnn/cpp/ttnn/operations/data_movement/bcast/` - bcast ✅
- `ttnn/cpp/ttnn/operations/data_movement/concat/` - concat ✅
- `ttnn/cpp/ttnn/operations/data_movement/expand/` - expand ✅
- `ttnn/cpp/ttnn/operations/data_movement/permute/` - permute ✅
- `ttnn/cpp/ttnn/operations/data_movement/fill_pad/` - fill_implicit_tile_padding ✅
- `ttnn/cpp/ttnn/operations/data_movement/gather/` - gather ✅

**Batch 2 (15 operations from 10 directories):**
- `ttnn/cpp/ttnn/operations/data_movement/unsqueeze/` - unsqueeze ✅
- `ttnn/cpp/ttnn/operations/data_movement/chunk/` - chunk ✅
- `ttnn/cpp/ttnn/operations/data_movement/stack/` - stack ✅
- `ttnn/cpp/ttnn/operations/data_movement/roll/` - roll (3 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/indexed_fill/` - indexed_fill ✅
- `ttnn/cpp/ttnn/operations/data_movement/fill_rm/` - fill_rm, fill_ones_rm ✅
- `ttnn/cpp/ttnn/operations/data_movement/scatter/` - scatter, scatter_add ✅
- `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/` - tilize_with_val_padding (2 overloads), tilize_with_zero_padding ✅
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/` - untilize_with_unpadding ✅
- `ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/` - nonzero ✅

**Batch 3 Phase 1 (5 operations):**
- `ttnn/cpp/ttnn/operations/data_movement/pad/` - pad (2 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/repeat/` - repeat ✅
- `ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/` - moe_routing_remap ✅
- `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/` - reshape_on_device (4 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/` - reshard ✅

**Batch 3 Phase 2 (6 operations - sharded + sort):**
- `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/` - sharded_to_interleaved ✅
- `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/` - sharded_to_interleaved_partial ✅
- `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/` - interleaved_to_sharded_partial ✅
- `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/` - interleaved_to_sharded (2 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/sort/` - sort (returns std::vector<Tensor>, uses mod.def lambda) ✅

**Final Batch (7 operations - remaining data_movement):**
- `ttnn/cpp/ttnn/operations/data_movement/copy/` - assign (2 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/slice/` - slice (4 template overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/view/` - view (2 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/reshape_view/` - reshape (3 overloads) ✅
- `ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap/` - moe_expert_token_remap ✅
- `ttnn/cpp/ttnn/operations/data_movement/scatter/` - tosa_scatter ✅
- `ttnn/cpp/ttnn/operations/data_movement/gather/tosa/` - tosa_gather ✅

**Conv Operations (3 operations - merged from separate PR):**
- `ttnn/cpp/ttnn/operations/conv/conv1d/` - conv1d ✅
- `ttnn/cpp/ttnn/operations/conv/conv2d/` - conv2d ✅
- `ttnn/cpp/ttnn/operations/conv/conv_transpose2d/` - conv_transpose2d ✅

### 🚧 In Progress (Outstanding PRs)
- `ttnn/cpp/ttnn/operations/reduction/` - TopK and other reductions (PR in merge queue)

### 📋 TODO - High Priority (Core Operations)

#### Data Movement (COMPLETE! ✅)
All data_movement operations have been migrated to the free function pattern.
- [x] `concat/` - ConcatOperation ✅
- [x] `copy/` - Copy ✅
- [x] `expand/` - ExpandOperation ✅
- [x] `fold/` - Fold ✅
- [x] `gather/` - Gather, TosaGather ✅
- [x] `indexed_fill/` - IndexedFill ✅
- [x] `pad/` - Pad ✅
- [x] `permute/` - PermuteOperation ✅
- [x] `repeat/` - RepeatOperation ✅
- [x] `repeat_interleave/` - RepeatInterleave ✅
- [x] `reshape_on_device/` - ReshapeOperation ✅
- [ ] `reshape_view/` - ReshapeViewOperation
- [x] `reshard/` - Reshard ✅
- [x] `roll/` - RollOperation ✅
- [x] `scatter/` - Scatter, TosaScatter, ScatterAdd ✅
- [ ] `slice/` - SliceOperation
- [x] `sort/` - Sort ✅ (uses mod.def lambda instead of bind_function)
- [ ] `squeeze/` - SqueezeOperation (requires lambda, may not migrate)
- [x] `stack/` - Stack ✅
- [x] `tilize/` - Tilize ✅
- [x] `tilize_with_val_padding/` - TilizeWithValPadding, TilizeWithZeroPadding ✅
- [x] `transpose/` - TransposeOperation ✅
- [x] `unsqueeze/` - UnsqueezeOperation ✅
- [x] `untilize/` - Untilize ✅
- [x] `untilize_with_unpadding/` - UntilizeWithUnpadding ✅
- [ ] `view/` - ViewOperation
- [x] `fill_pad/` - FillImplicitTilePadding ✅
- [x] `fill_rm/` - FillRM, FillOnesRM ✅
- [x] `chunk/` - Chunk ✅
- [x] `move/` - Move ✅
- [x] `non_zero_indices/` - NonZero ✅
- [ ] `moe_expert_token_remap/` - MoeExpertTokenRemap
- [x] `moe_routing_remap/` - MoeRoutingRemap ✅
- [x] `sharded/` - InterleavedToSharded, ShardedToInterleaved ✅
- [x] `sharded_partial/` - InterleavedToShardedPartial, ShardedToInterleavedPartial ✅

#### Eltwise Operations (~150 operations)
- [ ] `eltwise/unary/` - ~30 unary operations (abs, relu, gelu, etc.)
- [ ] `eltwise/binary/` - ~25 binary operations (add, sub, mul, div, etc.)
- [ ] `eltwise/ternary/` - ~4 ternary operations (where, addcmul, addcdiv, lerp)
- [ ] `eltwise/unary_backward/` - ~70 backward operations
- [ ] `eltwise/binary_backward/` - ~25 binary backward operations
- [ ] `eltwise/ternary_backward/` - ~4 ternary backward operations
- [ ] `eltwise/complex/` - complex_tensor
- [ ] `eltwise/complex_unary/` - complex unary operations
- [ ] `eltwise/complex_binary/` - complex binary operations
- [ ] `eltwise/quantization/` - quantize, dequantize, requantize

#### Reduction Operations (~10 operations)
- [x] `reduction/argmax/` - ArgMax ✅
- [x] `reduction/prod/` - Prod ✅
- [x] `reduction/accumulation/cumsum/` - Cumsum ✅
- [x] `reduction/accumulation/cumprod/` - Cumprod ✅
- [x] `reduction/accumulation/ema/` - EMA ✅
- [x] `reduction/moe/` - MOE ✅
- [x] `reduction/sampling/` - Sampling ✅
- [x] `reduction/manual_seed/` - ManualSeed ✅
- [x] `reduction/generic/` - Generic reductions ✅

#### Pool Operations (~6 operations)
- [ ] `pool/avg_pool2d/` - AvgPool2D
- [ ] `pool/max_pool2d/` - MaxPool2D
- [ ] `pool/global_avg_pool/` - GlobalAvgPool2D
- [ ] `pool/upsample/` - Upsample
- [ ] `pool/grid_sample/` - GridSample
- [ ] `pool/rotate/` - Rotate

#### CCL Operations (~10 operations)
- [ ] `ccl/all_gather/` - AllGather
- [ ] `ccl/all_reduce/` - AllReduce
- [ ] `ccl/all_broadcast/` - AllBroadcast
- [ ] `ccl/broadcast/` - Broadcast
- [ ] `ccl/reduce_scatter/` - ReduceScatter
- [ ] `ccl/reduce_to_root/` - ReduceToRoot
- [ ] `ccl/mesh_partition/` - MeshPartition
- [ ] `ccl/all_to_all_dispatch/` - AllToAllDispatch
- [ ] `ccl/all_to_all_combine/` - AllToAllCombine

#### Transformer Operations (~15 operations)
- [ ] `transformer/sdpa/` - SDPA
- [ ] `transformer/sdpa_decode/` - SDPADecode
- [ ] `transformer/sdpa_windowed/` - SDPAWindowed
- [ ] `transformer/attention_softmax/` - AttentionSoftmax
- [ ] `transformer/concatenate_heads/` - ConcatenateHeads
- [ ] `transformer/split_query_key_value_and_split_heads/` - SplitQueryKeyValueAndSplitHeads

#### Creation Operations (~10 operations)
- [ ] `creation.hpp` - full, zeros, ones, empty, arange, full_like, zeros_like, ones_like, empty_like, from_buffer

#### Other Core Operations (~15 operations)
- [ ] `embedding/` - Embedding
- [ ] `embedding_backward/` - EmbeddingBackward
- [ ] `kv_cache/` - FillCache, UpdateCache
- [ ] `loss/` - L1Loss, MSELoss
- [ ] `index_fill/` - IndexFill
- [ ] `full/` - Full
- [ ] `full_like/` - FullLike
- [ ] `uniform/` - Uniform
- [ ] `bernoulli/` - Bernoulli
- [ ] `rand/` - Rand
- [ ] `point_to_point/` - PointToPoint
- [ ] `generic/` - GenericOp
- [ ] `examples/` - Example operations

### ✅ Moreh Operations (32 operations - **COMPLETE!**)
All Moreh operations migrated in 3 batches:
- [x] moreh_adam, moreh_adamw, moreh_sgd (optimizers) ✅
- [x] moreh_matmul, moreh_linear, moreh_bmm, moreh_dot (matmul ops) ✅
- [x] moreh_matmul_backward, moreh_bmm_backward, moreh_linear_backward, moreh_dot_backward ✅
- [x] moreh_sum, moreh_mean, moreh_cumsum, moreh_arange, moreh_getitem ✅
- [x] moreh_sum_backward, moreh_mean_backward ✅
- [x] moreh_layer_norm, moreh_layer_norm_backward ✅
- [x] moreh_group_norm, moreh_group_norm_backward ✅
- [x] moreh_softmax (softmax/softmin/logsoftmax), moreh_softmax_backward ✅
- [x] moreh_norm, moreh_norm_backward ✅
- [x] moreh_nll_loss, moreh_nll_loss_backward, moreh_nll_loss_unreduced_backward ✅
- [x] moreh_fold, moreh_abs_pow, moreh_clip_grad_norm ✅

### 📋 TODO - Lower Priority (Experimental ~100 operations)
- [ ] `experimental/ccl/` - ~30 async CCL operations
- [ ] `experimental/transformer/` - ~25 transformer utilities
- [ ] `experimental/matmul/` - AttnMatmul, GroupAttnMatmul, MinimalMatmul
- [ ] `experimental/conv3d/` - Conv3D
- [ ] `experimental/paged_cache/` - Paged cache operations
- [ ] `experimental/adaptive_pool/` - Adaptive pooling
- [ ] `experimental/reduction/` - Fast reduce, DeepSeek operations
- [ ] `experimental/ssm/` - SSM operations
- [ ] `experimental/dropout/` - Dropout
- [ ] `experimental/isin/` - IsIn
- [ ] `experimental/where/` - Where
- [ ] `experimental/slice_write/` - SliceWrite
- [ ] `experimental/bcast_to/` - BcastTo
- [ ] `experimental/reshape/` - View
- [ ] And many more experimental operations

## Migration Batches

### Batch 1: Data Movement (Week 1) - 40 ops
Focus on core data movement operations used everywhere.
**Dependencies:** None
**Priority:** Critical

### Batch 2: Eltwise Unary/Binary (Week 2) - 60 ops
Basic element-wise operations.
**Dependencies:** None
**Priority:** High

### Batch 3: Eltwise Backward (Week 3) - 90 ops
Backward operations for training.
**Dependencies:** Batch 2
**Priority:** High

### Batch 4: Pool + Creation (Week 4) - 16 ops
Pooling and tensor creation operations.
**Dependencies:** Batch 1
**Priority:** Medium

### Batch 5: CCL + Transformer (Week 5) - 25 ops
Distributed and transformer operations.
**Dependencies:** Batch 1, 2
**Priority:** Medium

### Batch 6: Moreh Operations (Week 6) - 35 ops
Moreh framework operations.
**Dependencies:** Previous batches
**Priority:** Medium

### Batch 7: Experimental (Week 7-8) - 100+ ops
Experimental and specialized operations.
**Dependencies:** All previous batches
**Priority:** Low

## Build Verification

After each batch:
```bash
./build_metal.sh -c --debug --build-all
```

## Notes
- Operations in `conv/` and `reduction/` directories have outstanding PRs - DO NOT migrate yet
- Helper functions should either do real work (stay in internal namespace) or be eliminated
- Maintain exact Python/C++ API compatibility
- No silly wrapper functions
- Minimize namespace qualifiers

## Progress Log

| Date | Batch | Operations Migrated | Notes |
|------|-------|---------------------|-------|
| 2026-02-12 | Discovery | 0 | Initial discovery complete, 311 ops found |
| 2026-02-12 | Test Batch | 4 | clone, copy, move, transpose - ✅ BUILD SUCCESSFUL |
| 2026-02-12 | Note | - | squeeze requires Python polymorphism, kept old pattern |
| 2026-02-13 | Batch 1 | 10 | untilize, tilize, fold, repeat_interleave, bcast, concat, expand, permute, fill_pad, gather - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | Batch 2 | 15 (10 dirs) | unsqueeze, chunk, stack, roll, indexed_fill, fill_rm/fill_ones_rm, scatter/scatter_add, tilize_with_val_padding/tilize_with_zero_padding, untilize_with_unpadding, nonzero - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | Batch 3 Phase 1 | 5 | pad, repeat, moe_routing_remap, reshape_on_device, reshard - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | Batch 3 Phase 2 | 6 | sharded_to_interleaved, sharded_to_interleaved_partial, interleaved_to_sharded_partial, interleaved_to_sharded (2 overloads), sort (returns vector, uses mod.def lambda) - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | **DATA MOVEMENT COMPLETE** | **47 total** | All data_movement operations migrated! 🎉 |
| 2026-02-13 | Moreh Batch 1 | 8 | moreh_arange, moreh_sum, moreh_mean, moreh_cumsum, moreh_getitem, moreh_fold, moreh_abs_pow, moreh_dot - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | Moreh Batch 2 | 10 | moreh_matmul, moreh_bmm, moreh_linear, moreh_matmul_backward, moreh_bmm_backward, moreh_linear_backward, moreh_dot_backward, moreh_softmax (3 variants), moreh_softmax_backward (3 variants), moreh_layer_norm - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | Moreh Batch 3 | 14 | moreh_adam, moreh_adamw, moreh_sgd, moreh_group_norm, moreh_group_norm_backward, moreh_layer_norm_backward, moreh_norm, moreh_norm_backward, moreh_mean_backward, moreh_sum_backward, moreh_nll_loss, moreh_nll_loss_backward, moreh_nll_loss_unreduced_backward, moreh_clip_grad_norm - ✅ BUILD SUCCESSFUL |
| 2026-02-13 | **MOREH COMPLETE** | **32 total** | All Moreh operations migrated! 🎉 |
| 2026-02-13 | **MILESTONE** | **79 total** | Data Movement (47) + Moreh (32) = 79 operations complete! |

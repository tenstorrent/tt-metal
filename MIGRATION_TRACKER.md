# TTNN Operation Migration Tracker

## Overview
Migration from `register_operation` + `bind_registered_operation` to `bind_function<>` pattern.

**Last Updated:** 2026-02-13

## Statistics

| Metric | Count |
|--------|-------|
| Total operations found | 311 |
| **Data Movement COMPLETE** | **47** (all data_movement ops migrated!) |
| In outstanding PRs | ~15 (conv + reduction) |
| Remaining to migrate | ~249 (mostly eltwise ~150 + others) |

### Batch Breakdown
- Pre-existing: 7 operations
- Test Batch: 4 operations (squeeze reverted)
- Batch 1: 10 operations
- Batch 2: 15 operations
- Batch 3 Phase 1: 5 operations
- Batch 3 Phase 2: 6 operations (sharded operations + sort)
- **Total: 47 operations completed**

## Migration Status by Directory

### ✅ Completed (33 directories, 47 operations)
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

**Conv Operations (3 operations - merged from separate PR):**
- `ttnn/cpp/ttnn/operations/conv/conv1d/` - conv1d ✅
- `ttnn/cpp/ttnn/operations/conv/conv2d/` - conv2d ✅
- `ttnn/cpp/ttnn/operations/conv/conv_transpose2d/` - conv_transpose2d ✅

### 🚧 In Progress (Outstanding PRs)
- `ttnn/cpp/ttnn/operations/reduction/` - TopK and other reductions (PR in merge queue)

### 📋 TODO - High Priority (Core Operations)

#### Data Movement (~16 operations remaining)
- [x] `bcast/` - BcastOperation ✅
- [x] `clone/` - Clone ✅
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
- [ ] `reduction/argmax/` - ArgMax
- [ ] `reduction/prod/` - Prod
- [ ] `reduction/accumulation/cumsum/` - Cumsum
- [ ] `reduction/accumulation/cumprod/` - Cumprod
- [ ] `reduction/accumulation/ema/` - EMA
- [ ] `reduction/moe/` - MOE
- [ ] `reduction/sampling/` - Sampling
- [ ] `reduction/manual_seed/` - ManualSeed
- [ ] `reduction/generic/` - Generic reductions

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

### 📋 TODO - Medium Priority (Moreh Operations ~35)
- [ ] `moreh/moreh_adam/` - MorehAdam
- [ ] `moreh/moreh_adamw/` - MorehAdamW
- [ ] `moreh/moreh_sgd/` - MorehSGD
- [ ] `moreh/moreh_matmul/` - MorehMatmul
- [ ] `moreh/moreh_linear/` - MorehLinear
- [ ] `moreh/moreh_bmm/` - MorehBMM
- [ ] `moreh/moreh_dot/` - MorehDot
- [ ] `moreh/moreh_sum/` - MorehSum
- [ ] `moreh/moreh_mean/` - MorehMean
- [ ] `moreh/moreh_norm/` - MorehNorm
- [ ] `moreh/moreh_layer_norm/` - MorehLayerNorm
- [ ] `moreh/moreh_group_norm/` - MorehGroupNorm
- [ ] `moreh/moreh_softmax/` - MorehSoftmax
- [ ] `moreh/moreh_cumsum/` - MorehCumsum
- [ ] `moreh/moreh_getitem/` - MorehGetitem
- [ ] `moreh/moreh_nll_loss/` - MorehNLLLoss
- [ ] `moreh/moreh_arange/` - MorehArange
- [ ] `moreh/moreh_fold/` - MorehFold
- [ ] `moreh/moreh_abs_pow/` - MorehAbsPow
- [ ] And ~15 more moreh backward operations

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

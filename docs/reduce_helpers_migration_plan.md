# Reduce Helpers Library Migration Plan

## 1. Library Overview

### `reduce_helpers.hpp` - Unified Reduce Interface

Single unified reduce function that handles all reduction patterns:
- **REDUCE_ROW**: Reduces W dimension, outputs Ht tiles per batch
- **REDUCE_COL**: Reduces H dimension, outputs Wt tiles per batch (with DEST chunking)
- **REDUCE_SCALAR**: Reduces both H and W, outputs 1 tile per batch

**Input Modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `STREAMING` (default) | `cb_wait_front`/`cb_pop_front` per tile | Standard dataflow |
| `PRELOADED` | All tiles in CB, accessed via indexing | Sharded kernels, multi-pass |

**All Modes Implemented:**
| Reduce Dimension | STREAMING | PRELOADED |
|------------------|-----------|-----------|
| REDUCE_ROW       | ✅        | ✅        |
| REDUCE_COL       | ✅        | ✅        |
| REDUCE_SCALAR    | ✅        | ✅        |

**PRELOADED Features:**
- `input_stride` parameter for non-contiguous layouts
- Bulk reserve/push for better performance
- Row-major tile indexing with batch offset tracking

---

## 2. Migration Progress

### ✅ Phase 1: Core Library + Basic Reduces (COMPLETE)

Commit: `f60fc7a314` (p1 reduce)

| Kernel | File | Status |
|--------|------|--------|
| reduce_h | `reduction/generic/.../reduce_h.cpp` | ✅ Migrated |
| reduce_hw | `reduction/generic/.../reduce_hw.cpp` | ✅ Migrated |
| reduce_w | `reduction/generic/.../reduce_w.cpp` | ✅ Migrated |

### ✅ Phase 2: LayerNorm Kernels (COMPLETE)

Commits: `291e25d722` (p2), `19a92c9b72` (p3)

| Kernel | File | Status |
|--------|------|--------|
| layernorm_sharded | `normalization/layernorm/.../layernorm_sharded.cpp` | ✅ Migrated |
| layernorm_large_tensor | `normalization/layernorm/.../layernorm_large_tensor.cpp` | ✅ Migrated |
| layernorm_sharded_pre_allgather | `normalization/layernorm/.../layernorm_sharded_pre_allgather.cpp` | ✅ Migrated |
| layernorm_sharded_post_allgather | `normalization/layernorm/.../layernorm_sharded_post_allgather.cpp` | ✅ Migrated |

### ✅ Phase 3: GroupNorm Kernels (COMPLETE)

Commits: `fb8cf5c86a` (migrate GN), `9ec230add1` (gn sharded), `19a92c9b72` (p3)

| Kernel | File | Status |
|--------|------|--------|
| groupnorm | `normalization/groupnorm/.../groupnorm.cpp` | ✅ Migrated |
| groupnorm_sharded_v2 | `normalization/groupnorm/.../groupnorm_sharded_v2.cpp` | ✅ Migrated |

**Note:** Despite initial concerns about complexity (multi-pass reduce with interleaved ops), these kernels were successfully migrated using the library's `init`/`uninit` template parameters for fine-grained control.

---

## 3. Remaining Kernel Candidates

### Priority 1: Softmax Operations

Dual reductions (MAX + SUM) requiring two passes:

| Kernel | File | Reduce Type | Input Mode | Notes |
|--------|------|-------------|------------|-------|
| Softmax | `normalization/softmax/.../softmax.cpp` | ROW | Streaming | MAX then SUM reduces |
| Softmax Sharded | `normalization/softmax/.../softmax_sharded.cpp` | ROW | Preloaded | Uses indexed access |
| Softmax Large Tensor | `normalization/softmax/.../softmax_large_tensor.cpp` | ROW | Streaming | Multi-pass chunking |

**Challenge:** MAX reduce → exp/sub → SUM reduce with intermediate transformations. Can use library's `init=false`/`uninit=false` for multi-pass control.

### Priority 2: Moreh Operations

Large set of operations with consistent patterns:

| Category | Files | Reduce Types |
|----------|-------|--------------|
| Moreh Dot | `moreh_dot.cpp` | SCALAR |
| Moreh Softmax | `moreh_softmax_w.cpp`, `moreh_softmax_h.cpp`, `*_large.cpp` | ROW, COL |
| Moreh Softmax Backward | `moreh_softmax_backward_*.cpp` | ROW, COL |
| Moreh Layer Norm | `moreh_layer_norm_small_kernel.cpp`, `*_large_kernel.cpp` | SCALAR |
| Moreh Layer Norm Backward | `moreh_layer_norm_backward_*.cpp` | SCALAR |
| Moreh Mean | `moreh_mean_h.cpp` | COL |
| Moreh Norm | `moreh_norm_w_kernel.cpp` | ROW |
| Moreh Bias Backward | `moreh_bias_backward_*.cpp` | ROW, SCALAR |

**Strategy:** Moreh operations use `moreh_common.hpp` with specialized helpers. Options:
1. Migrate `moreh_common.hpp` to use `reduce_helpers.hpp` internally
2. Migrate individual kernels directly

### Priority 3: Complex/Specialized Operations

These have specialized patterns - evaluate case-by-case:

| Kernel | File | Notes |
|--------|------|-------|
| SDPA Decode | `transformer/sdpa_decode/.../compute_common.hpp` | Custom granularity, mixed ROW/COL |
| SDPA | `transformer/sdpa/.../compute_common.hpp` | Block-based with custom chunk sizes |
| MOE | `reduction/moe/.../moe.cpp` | Interspersed with topk operations |
| Pool 2D | `pool/generic/.../compute_pool_2d.cpp` | Custom pool patterns with indices |
| CCL Reduce | `ccl/reduce_to_root/.../compute_kernel.cpp` | Multi-pass all-reduce |
| DeepSeek Grouped Gate | `experimental/reduction/deepseek_grouped_gate/...` | Expert gating with topk |
| Welford GroupNorm | `normalization/groupnorm/.../welford_groupnorm.cpp` | Welford algorithm |

---

## 4. Next Steps

### Phase 4: Softmax Migration

1. Analyze softmax kernel patterns in detail
2. Use library with `init`/`uninit` control for multi-pass reduces
3. Test with existing softmax test suite

### Phase 5: Moreh Operations Migration

1. Evaluate `moreh_common.hpp` integration approach
2. Start with simple kernels (`moreh_dot.cpp`)
3. Migrate remaining moreh kernels as needed

### Phase 6: Complex Operations (Case-by-Case)

- Evaluate each operation individually
- May need library extensions or keep specialized implementations

---

## 5. Benefits of Migration

1. **Code Reduction:** Eliminates repeated DEST register management boilerplate
2. **Consistency:** Unified reduce patterns across codebase
3. **Maintainability:** Single place to fix bugs or optimize
4. **Flexibility:** Easy to switch between STREAMING/PRELOADED modes
5. **Auto DEST Limit:** Automatic detection via `dest_helpers.hpp`
6. **Fine-grained Control:** `init`/`uninit` template params for multi-pass operations

---

## 6. Files Reference

### Library Files
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` - Main reduce library
- `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` - DEST limit detection

### Migrated Kernels (Reference Implementations)
```
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_hw.cpp
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp
```

### Next Migration Candidates
```
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_sharded.cpp
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_large_tensor.cpp
ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp
ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp
ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h.cpp
```

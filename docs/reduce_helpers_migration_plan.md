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

### Priority 2: Moreh Operations - ⚠️ BLOCKED

**Status: Cannot migrate with current library API**

Moreh operations use `moreh_common.hpp` with specialized helpers like `reduce_tile_to_cb<>()`. After detailed analysis, **ALL Moreh kernels are blocked** from migration due to a fundamental architectural incompatibility.

#### Blocking Issue: Manual Accumulation Pattern

All Moreh kernels use a **manual accumulation pattern** that requires injecting previous accumulator values into DST registers between `tile_regs_acquire()` and `reduce_tile()`:

```cpp
// Common Moreh pattern (e.g., moreh_dot.cpp:40-64, moreh_mean_h.cpp:91-101, moreh_bias_backward:86-107)
for each block:
  tile_regs_acquire()
  if (enable_reload):
    copy_tile(cb_accumulator → DST[0])  // ← INJECT previous accumulator
  reduce_init(...)
  reduce_tile(...)  // ← Adds to DST[0]
  reduce_uninit()
  pack_tile(DST[0] → cb_accumulator)
  tile_regs_release()
```

**Library's pattern** (ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp:239-256):
```cpp
tile_regs_acquire()  // ← Can't inject between here and reduce_tile
for each tile:
  reduce_tile(...)
tile_regs_commit()
pack_tile()
tile_regs_release()
```

#### Detailed Assessment by Category

| Category | Files | Pattern | Can Migrate? | Blocker |
|----------|-------|---------|--------------|---------|
| **Moreh Dot** | `moreh_dot.cpp` | Per-tile reduce with cross-block accumulation | ❌ No | Needs reload between acquire/reduce |
| **Moreh Softmax** | `moreh_softmax_w.cpp`<br>`moreh_softmax_h.cpp` | MAX reduce (with reload for masking)<br>+ SUM reduce on exp values | ❌ No | MAX reduce uses reload pattern (lines 51-58) |
| **Moreh Softmax Backward** | `moreh_softmax_backward_w.cpp`<br>`moreh_softmax_backward_h.cpp` | Uses `reduce_tile_to_cb` helper<br>Multi-tile streaming reduces | ⚠️ Partial | Could migrate non-accumulating reduces, but relies on moreh_common.hpp |
| **Moreh Layer Norm** | `moreh_layer_norm_small_kernel.cpp` | Complex multi-pass:<br>• Sum[x] with manual accumulation (lines 78-161)<br>• Reduce to E[x] (lines 172-174)<br>• Sum[(x-E[x])²] with manual accumulation (lines 251-298)<br>• Reduce to Var[x] (lines 309-311) | ❌ No | Manual accumulation for sums;<br>individual reduces are trivial (3 lines) |
| **Moreh Layer Norm Backward** | `moreh_layer_norm_backward_input_grad_small_kernel.cpp` | Similar to forward: manual accumulation + simple reduces | ❌ No | Same as forward |
| **Moreh Mean** | `moreh_mean_h.cpp` | Reduce Ht-1 tiles → accumulate<br>Reload accumulator → reduce last tile | ❌ No | Reload pattern (lines 94-96) |
| **Moreh Norm** | `moreh_norm_w_kernel.cpp` | Single reduce (lines 134-136) embedded in complex power/log operations | ⚠️ Minimal | Reduce is 3 lines; migration overhead not worth it |
| **Moreh Bias Backward** | `moreh_bias_backward_multi_core_h.cpp`<br>`moreh_bias_backward_single_core_hw.cpp` | Per-tile reduce with cross-tile accumulation + reload | ❌ No | Reload pattern (lines 87-95) |

#### Why Moreh Common Helpers Don't Help

The `moreh_common.hpp` provides `reduce_tile_to_cb<>()` which handles **multi-tile streaming reduces** (lines 593-626):
- Accepts `size` parameter to reduce multiple tiles
- Manages `reduce_init`, loop of `reduce_tile`, `reduce_uninit`
- BUT: Does NOT support reloading previous accumulators

Even if we migrated `moreh_common.hpp` to use `reduce_helpers.hpp`, the accumulation pattern issue remains.

#### Required Library Changes

To unblock Moreh operations, `reduce_helpers.hpp` needs:

1. **Option 1: External DST Management**
   ```cpp
   template <bool skip_acquire = false, bool skip_release = false>
   void reduce(...);
   ```
   Allow kernels to manage `tile_regs_acquire/release` externally

2. **Option 2: Accumulator Injection**
   ```cpp
   template <bool load_accumulator = false>
   void reduce(uint32_t accumulator_cb = 0, ...);
   ```
   Support loading previous accumulator before reduce

3. **Option 3: Specialized Accumulating Reduce**
   ```cpp
   void reduce_accumulating(
       uint32_t icb, uint32_t scaler_cb, uint32_t accumulator_cb,
       uint32_t ocb, uint32_t num_iterations, ...);
   ```
   Dedicated function for multi-pass accumulation patterns

**Recommendation:** Until library API is extended, Moreh operations must remain as-is using `moreh_common.hpp` helpers.

### Priority 3: Complex/Specialized Operations

| Kernel | File | Notes |
|--------|------|-------|
| SDPA Decode | `transformer/sdpa_decode/.../compute_common.hpp` | Preloaded indexed pattern, valid candidate |

---

## 4. Next Steps

### Phase 4: Softmax Migration

1. Analyze softmax kernel patterns in detail
2. Use library with `init`/`uninit` control for multi-pass reduces
3. Test with existing softmax test suite

### ~~Phase 5: Moreh Operations Migration~~ - ⚠️ BLOCKED

**Status:** Cannot proceed until `reduce_helpers.hpp` API is extended to support accumulation patterns.

**Blockers identified:**
1. All Moreh kernels require manual accumulator reload between `tile_regs_acquire()` and `reduce_tile()`
2. Current library manages full DST lifecycle without injection points
3. `moreh_common.hpp` helpers also don't support this pattern

**Next actions:**
1. ~~Evaluate `moreh_common.hpp` integration approach~~ → Not viable, same limitation
2. ~~Start with simple kernels (`moreh_dot.cpp`)~~ → Blocked (see Priority 2 analysis)
3. **DECISION NEEDED:** Extend library API (see 3 options in Priority 2) OR keep Moreh operations using existing patterns

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

---

## 7. Complete Kernel Audit

**Summary:** 73 files matched grep for "reduce_tile", but many only include headers or mention it in comments. Actual compute kernels using `reduce_tile()`:

| Category | Count | Status |
|----------|-------|--------|
| Already Migrated | 10 | ✅ Done |
| Easy Migration (distributed norms) | 7 | 🟢 Need `skip_dst_management` |
| Softmax | 3 | 🟡 Need `skip_dst_management` |
| Moreh | 26 | 🔴 Need `load_accumulator` |
| Complex/Specialized | 5 | 🟠 Case-by-case |
| Test Kernels | 10 | 🟤 Keep as reference |
| Demo-specific | 1 | 📁 Low priority |
| **Total** | **62** | |

### Category 1: ✅ ALREADY MIGRATED (10 files)

These kernels already use `reduce_helpers.hpp`:

| File | Status |
|------|--------|
| `reduction/generic/.../reduce_h.cpp` | ✅ Uses library |
| `reduction/generic/.../reduce_hw.cpp` | ✅ Uses library |
| `reduction/generic/.../reduce_w.cpp` | ✅ Uses library |
| `normalization/layernorm/.../layernorm_sharded.cpp` | ✅ Uses library |
| `normalization/layernorm/.../layernorm_large_tensor.cpp` | ✅ Uses library |
| `normalization/layernorm/.../layernorm_sharded_pre_allgather.cpp` | ✅ Uses library |
| `normalization/layernorm/.../layernorm_sharded_post_allgather.cpp` | ✅ Uses library |
| `normalization/groupnorm/.../groupnorm.cpp` | ✅ Uses library |
| `normalization/groupnorm/.../groupnorm_sharded_v2.cpp` | ✅ Uses library |
| `reduction/tilize_untilize/.../tilize_untilize_compute.cpp` | ✅ Uses library |

---

### Category 2: 🟢 EASY MIGRATION - Standard Row Reduce (Preloaded) (7 files)

**Pattern:** Simple REDUCE_ROW with preloaded tiles, external DST management.

**Library Support Required:**
- `skip_dst_management` template parameter to allow kernel to manage `tile_regs_acquire/release` externally

| File | Pattern | Assessment |
|------|---------|------------|
| `layernorm_distributed/.../layernorm_pre_allgather.cpp` | REDUCE_ROW, preloaded Wt tiles, two sequential reduces (sum(x²), sum(x)) | ✅ **Can migrate** - Use library 2x with `init=false`/`uninit=false` for second call. Need external DST management. |
| `layernorm_distributed/.../layernorm_pre_allgather_2d.cpp` | Same as above + merge core logic | ✅ **Can migrate** - Same pattern, merge core uses add_tiles (unrelated) |
| `layernorm_distributed/.../layernorm_post_allgather.cpp` | Two reduces: sum(x²) cols 0,2,4.. then sum(x) cols 1,3,5.. to different DST | ⚠️ **Partial** - Non-contiguous column access, library doesn't support stride patterns |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | REDUCE_ROW, preloaded, single reduce sum(x²) | ✅ **Can migrate** - Direct match with external DST management |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather_2d.cpp` | Same as above + merge core | ✅ **Can migrate** - Same pattern |
| `rmsnorm_distributed/.../rmsnorm_post_allgather.cpp` | Similar to layernorm post | ⚠️ **Partial** - Same non-contiguous issue |
| `sampling/.../sampling.cpp` | REDUCE_ROW in helper function `reduce_c()` | ✅ **Can migrate** - Standard preloaded pattern |

**Library Extension Needed:**
```cpp
template <..., bool skip_dst_management = false>
void reduce(...);
```
When `skip_dst_management=true`, library skips `tile_regs_acquire/commit/wait/release` and lets kernel handle it.

---

### Category 3: 🟡 MEDIUM MIGRATION - Softmax Pattern (3 files)

**Pattern:** Dual reduce (MAX + SUM) with intermediate exp/sub operations.

| File | Pattern | Assessment |
|------|---------|------------|
| `softmax/.../softmax.cpp` | Cumulative cb_wait_front, indexed access | ⚠️ **Medium** - Need PRELOADED mode + external DST |
| `softmax/.../softmax_sharded.cpp` | PRELOADED, single reduce_tile per row then uninit | ⚠️ **Medium** - Need single-batch, single-row variant |
| `softmax/.../softmax_large_tensor.cpp` | STREAMING single-tile reduce loop | ✅ **Easy** - Standard REDUCE_ROW streaming |

**Library Extension Needed:**
- Already have `init`/`uninit` params ✅
- Need `skip_dst_management` for external register control

---

### Category 4: 🔴 BLOCKED - Moreh Operations (26 files)

**Pattern:** Manual accumulator reload between `tile_regs_acquire()` and `reduce_tile()`.

All Moreh kernels blocked due to:
```cpp
tile_regs_acquire();
if (enable_reload):
  copy_tile(cb_accumulator → DST[0])  // INJECT previous accumulator
reduce_tile(...)  // Adds to DST[0]
pack_tile(DST[0] → cb_accumulator)
tile_regs_release();
```

**Files:**
- `moreh_dot.cpp`, `moreh_mean_h.cpp`, `moreh_sum_h.cpp`
- `moreh_softmax_w.cpp`, `moreh_softmax_h.cpp`, `moreh_softmax_w_large.cpp`, `moreh_softmax_h_large.cpp`
- `moreh_softmax_backward_*.cpp` (4 files)
- `moreh_layer_norm_*.cpp` (2 files)
- `moreh_layer_norm_backward_*.cpp` (3 files)
- `moreh_bias_backward_*.cpp` (2 files)
- `moreh_clip_grad_norm_step1_kernel.cpp`
- `moreh_norm_*.cpp` (4 files)
- `moreh_sum_nc_*.cpp` (2 reader files - program factory level)

**Library Extension Needed:**
```cpp
template <..., bool load_accumulator = false>
void reduce(..., uint32_t accumulator_cb = 0);
```
Or new function:
```cpp
void reduce_with_accumulator(uint32_t icb, uint32_t scaler_cb, uint32_t accumulator_cb, uint32_t ocb, ...);
```

---

### Category 5: 🟠 COMPLEX - Specialized Patterns (5 files)

**Pattern:** Custom reduction logic tightly integrated with other operations.

| File | Pattern | Assessment |
|------|---------|------------|
| `transformer/sdpa_decode/.../compute_common.hpp` | Preloaded indexed `r * cols + c` pattern with external DST | ✅ **Can migrate** - Standard PRELOADED pattern |
| `experimental/ccl/rms_allgather/.../rms_compute.cpp` | Multi-stage reduce with interleaved ops | ⚠️ **Partial** - Could migrate individual reduce calls |
| `experimental/transformer/fused_distributed_rmsnorm/.../rmsnorm_pre_allgather.cpp` | Multi-stage reduce pattern | ⚠️ **Partial** - Similar to distributed norm |
| `experimental/transformer/fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp` | Same | ⚠️ **Partial** |
| `experimental/ssm/hc_sum_reduce/.../ssm_1d_sum_reduce.cpp` | Single-tile REDUCE_COL + transpose | ✅ **Can migrate** - Simple pattern |

---

### Category 6: 🔵 TT-TRAIN Kernels

**No compute kernels use reduce_tile.** The grep matches were in dataflow/reader kernels (non-compute), which don't use this library.

---

### Category 7: 🟤 Test Kernels (10 files)

**Pattern:** Unit test kernels, low priority.

| File | Assessment |
|------|------------|
| `tests/tt_metal/.../reduce_h.cpp` | Test kernel - keep as reference |
| `tests/tt_metal/.../reduce_hw.cpp` | Test kernel |
| `tests/tt_metal/.../reduce_w.cpp` | Test kernel |
| `tests/tt_metal/.../softmax.cpp` | Test kernel |
| `tests/tt_metal/.../layernorm.cpp` | Test kernel |
| `tests/tt_metal/.../rmsnorm.cpp` | Test kernel |
| `tests/tt_metal/.../max_pool.cpp` | Test kernel |
| `tests/tt_metal/.../max_pool_multi_core.cpp` | Test kernel |
| `tests/tt_metal/.../test_reduce.cpp` | Test driver |

**Recommendation:** Leave as-is. Test kernels serve as LLK reference implementations.

---

### Category 8: 📁 Demo/Model Specific (1 file)

| File | Assessment |
|------|------------|
| `models/demos/deepseek_v3_b1/.../rmsnorm_compute.cpp` | Model-specific kernel | 🔍 **Investigate** - May follow standard pattern |

---

## 8. Library Extensions Summary

### Required Extensions for Broader Migration

| Extension | Priority | Unblocks |
|-----------|----------|----------|
| `skip_dst_management` template param | **HIGH** | 7 files (Cat 2) + 3 files (Cat 3) |
| `load_accumulator` support | **MEDIUM** | 26 files (Cat 4 - Moreh) |
| Non-contiguous column access (stride) | **LOW** | 2 files (post_allgather) |

### Proposed API Extension

```cpp
template <
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    ReduceInputMode input_mode = ReduceInputMode::STREAMING,
    bool init = true,
    bool uninit = true,
    bool enforce_fp32_accumulation = false,
    bool skip_dst_management = false,      // NEW: Let kernel manage tile_regs_*
    bool load_accumulator = false>         // NEW: Load previous accumulator into DST
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_batches,
    uint32_t row_chunk = 0,
    uint32_t input_stride = 0,
    uint32_t accumulator_cb = 0);          // NEW: CB for accumulator when load_accumulator=true
```

---

## 9. Migration Roadmap

### Phase 4: Easy Migrations (No Library Changes)
- `softmax_large_tensor.cpp` - Standard streaming pattern
- `ssm_1d_sum_reduce.cpp` - Simple single-tile reduce

### Phase 5: With `skip_dst_management` Extension
- `layernorm_pre_allgather.cpp`
- `layernorm_pre_allgather_2d.cpp`
- `rmsnorm_pre_allgather.cpp`
- `rmsnorm_pre_allgather_2d.cpp`
- `sampling.cpp`
- `softmax.cpp`
- `softmax_sharded.cpp`
- `sdpa_decode/compute_common.hpp` - Standard PRELOADED indexed pattern

### Phase 6: With `load_accumulator` Extension (Moreh)
- All 26 Moreh kernels become candidates

### Not Planned for Migration
- Test kernels (reference implementations)

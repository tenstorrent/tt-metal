# Reduce Helpers Library Migration Plan

## Status Summary (as of 2026-01-05)

**Scope:** `ttnn/cpp/ttnn/operations/` folder only

**Progress:** (26 kernel files with `reduce_tile<` in operations/)
- ✅ **21 files fully migrated** - Use only `compute_kernel_lib::reduce`
- 🔶 **3 files partially migrated** - Use both library AND raw `reduce_tile<`
- ⏳ **22 files blocked** - Need library extensions (accumulation, interleaved, stride patterns)

**Note:** 3 files previously marked "ready" are actually blocked (dual-DEST, accumulator reload patterns)

**Latest Achievement:** Library API finalized - TileShape, TileLayout, ReduceDataFormatReconfig, explicit template params (no macros)

---

## 1. Library Overview

### `reduce_helpers.hpp` - Unified Reduce Interface

**Location:** `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`

**Reduce Dimensions:**
- **REDUCE_ROW**: W dimension → outputs Ht tiles/batch
- **REDUCE_COL**: H dimension → outputs Wt tiles/batch
- **REDUCE_SCALAR**: H×W → outputs 1 tile/batch

**Input Modes:**
| Mode | Behavior | Use Case |
|------|----------|----------|
| STREAMING (default) | One-at-a-time, safe for any CB size | Simple cases, numerical precision |
| STREAMING_BATCHED | Waits all tiles per row/batch, indexed access, pops all | Optimal perf when tiles pre-loaded |
| PRELOADED | All tiles ready, indexed access, caller manages wait/pop | When tiles must persist in CB |
| PERSISTENT | All tiles ready, NO pop (reusable) | Multi-pass scenarios (softmax) |

**Key Features:**
- **TileShape struct**: `TileShape::grid(Ht, Wt, NC)`, `TileShape::row(Wt)`, `TileShape::single()`
- **TileLayout struct**: `TileLayout::contiguous()`, `TileLayout::with_row_stride(stride)`
- **ReduceDataFormatReconfig**: `NONE`, `INPUT`, `OUTPUT`, `BOTH` (default)
- **Explicit template params**: `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()` (no macros)
- **PostReduceOp lambda**: Custom post-reduce ops (e.g., `recip_tile` for softmax)
- **init/uninit control**: Fine-grained multi-pass support

### `dest_helpers.hpp` - DEST Register Utilities

**Location:** `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`

**Features:**
- `get_dest_limit()`: Auto-detect DEST capacity based on sync/accum mode
- `get_fp32_dest_acc_enabled()`: Detect FP32 accumulation from JIT headers
- `DEST_AUTO_LIMIT`: Constexpr default for current configuration

---

## 2. Migration Status

### ✅ Fully Migrated (21 files)

| Phase | Count | Kernels | Mode |
|-------|-------|---------|------|
| **P1: Basic Reduces** | 3 | reduce_h, reduce_hw, reduce_w | STREAMING_BATCHED |
| **P3: GroupNorm** | 1 | groupnorm | PRELOADED |
| **P4: Softmax** | 4 | softmax, softmax_sharded, softmax_large_tensor, tilize_untilize | PERSISTENT + PostReduceOp |
| **P5: Specialized** | 2 | deepseek_grouped_gate, moe | PRELOADED |
| **P6: Distributed Norms** | 5 | layernorm/rmsnorm_pre_allgather{,_2d} (4), experimental/rmsnorm_pre | STREAMING_BATCHED |
| **P7: Simple Ops** | 3 | sampling, ssm_1d_sum_reduce, sdpa_decode | STREAMING + PostReduceOp |
| **P8: RMSNorm Post-Allgather** | 2 | rmsnorm_post_allgather, experimental/rmsnorm_post_allgather | STREAMING_BATCHED |
| **P9: CCL RMS** | 1 | experimental/ccl/rms_allgather/rms_compute | PRELOADED + STREAMING |

**Key Achievements:**
- **P4:** Dual-pass MAX+SUM reduces with PERSISTENT mode + PostReduceOp lambdas
- **P6-P8:** STREAMING_BATCHED mode - bulk wait/pop, same efficiency as PRELOADED
- **Library API:** TileShape, TileLayout, ReduceDataFormatReconfig, explicit template params (no macros)

### 🔶 Partially Migrated (3 files)

Files using **both** `compute_kernel_lib::reduce` AND raw `reduce_tile<>`:

| File | Library Calls | Raw reduce_tile | Notes |
|------|---------------|-----------------|-------|
| groupnorm_sharded_v2.cpp | Yes (PRELOADED) | Yes (REDUCE_SCALAR) | Mixed usage in different code paths |
| layernorm_sharded.cpp | Yes (1 call) | Yes (3 calls) | Partial migration |
| layernorm_sharded_pre_allgather.cpp | Yes (2 calls) | Yes (1 call) | Partial migration |

### ⏳ Blocked - Non-Moreh (3 files - need library extensions)

| File | Pattern | Blocker |
|------|---------|---------|
| layernorm_large_tensor.cpp | Accumulator reload | Same as Moreh - inject accumulator before reduce |
| layernorm_sharded_post_allgather.cpp | Interleaved dual-DEST | `w % stats_tiles` alternates dst0/dst1 |
| layernorm_post_allgather.cpp (distributed) | Stride-2 dual-DEST | tiles 0,2,4→dst0, 1,3,5→dst1 |

### ⏳ Blocked - Moreh (19 files - need accumulation support)

**Moreh Operations (19 files):** Use manual accumulator reload pattern between `tile_regs_acquire()` and `reduce_tile()`:
- moreh_softmax (4) - h, w, h_large, w_large
- moreh_norm (4) - h, w, ord_other/h, ord_other/w
- moreh_layer_norm_backward (3) - gamma_beta_grad, input_grad_large, input_grad_small
- moreh_layer_norm (2) - small, large
- moreh_bias_backward (2) - single_core_hw, multi_core_h
- moreh_dot (1)
- moreh_mean (1) - h
- moreh_sum (1) - h
- moreh_clip_grad_norm (1) - step1

---

## 3. Blocker Details

### Moreh Operations (19 files)

**Issue:** Manual accumulator injection between `tile_regs_acquire()` and `reduce_tile()`

```cpp
// Moreh pattern - library doesn't support
tile_regs_acquire();
if (enable_reload):
  copy_tile(cb_accumulator → DST[0])  // ← Can't inject here
reduce_tile(...);  // Adds to DST[0]
pack_tile(DST[0] → cb_accumulator);
tile_regs_release();
```

**Solution Options:**
1. Add `load_accumulator` template parameter - load from CB before reduce
2. Add `skip_dst_management` for external register control
3. New `reduce_accumulating()` function - specialized for accumulation patterns

---

## 4. File Reference

### Library Files
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` - Unified reduce interface
- `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` - DEST register utilities

### Fully Migrated Kernels (21 files)
```
# P1: Basic Reduces (3 files)
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_{h,hw,w}.cpp

# P3: GroupNorm (1 file)
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp

# P4: Softmax + Tilize/Untilize (4 files)
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/{softmax,softmax_sharded,softmax_large_tensor}.cpp
ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp

# P5: Specialized (2 files)
ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp
ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp

# P6: Distributed Norms Pre-Allgather (5 files)
ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather{,_2d}.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather{,_2d}.cpp
ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp

# P7: Simple Ops (3 files)
ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp
ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/ssm_1d_sum_reduce.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp

# P8: RMSNorm Post-Allgather (2 files)
ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp
ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp

# P9: CCL RMS (1 file)
ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp
```

### Partially Migrated Kernels (3 files)
```
# Use both compute_kernel_lib::reduce AND raw reduce_tile<
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp
```

### Blocked - Non-Moreh (3 files)
```
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp
```

### Note: Helper Library (not a kernel)
```
# Uses reduce_tile< but is a utility header, not a kernel file
ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h
```

---

## 5. Key Takeaways

**Achieved:**
- **21 files fully migrated** - use only `compute_kernel_lib::reduce`
- **3 files partially migrated** - use both library AND raw `reduce_tile<`
- **Library API finalized:** TileShape, TileLayout, ReduceDataFormatReconfig, explicit template params
- Complex multi-pass reductions working (softmax, layernorm, groupnorm, distributed norms)

**Library Features:**
- `TileShape::grid(Ht, Wt, NC)` - self-documenting grid dimensions
- `TileLayout::with_row_stride(stride)` - non-contiguous layout support
- `ReduceDataFormatReconfig::{NONE,INPUT,OUTPUT,BOTH}` - fine-grained reconfig control
- PERSISTENT mode for multi-pass operations (softmax MAX+SUM pattern)
- PostReduceOp lambda for custom post-reduce operations

**Next Steps:**
- Complete migration of 3 partially migrated files (blocked patterns in remaining reduce_tile calls)
- Extend library for accumulation patterns to unblock 22 files (19 Moreh + 3 non-Moreh)

**Blocked (22 files):**
- 19 Moreh files need accumulation support (manual accumulator reload pattern)
- 3 non-Moreh files need dual-DEST or accumulator reload support
- Decision needed: extend library with `reduce_accumulating()` vs keep specialized implementations

**ROI:** 81% fully migrated (21/26), 92% using library (24/26)

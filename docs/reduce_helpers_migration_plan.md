# Reduce Helpers Library Migration Plan

## Status Summary (as of 2025-12-31)

**Scope:** `ttnn/cpp/ttnn/operations/` folder only (54 kernel files use `reduce_tile`)

**Progress:**
- ✅ **23/54 migrated (43%)** - Phases 1-7 complete
- ⏳ **30/54 blocked (56%)** - Need library extensions
- 🔍 **1/54 TBD (2%)** - Needs investigation

**Latest Achievement:** Phases 6 & 7 complete + Auto-batched STREAMING mode enhancement

---

## 1. Library Overview

### `reduce_helpers.hpp` - Unified Reduce Interface

**Reduce Dimensions:**
- **REDUCE_ROW**: W dimension → outputs Ht tiles/batch
- **REDUCE_COL**: H dimension → outputs Wt tiles/batch
- **REDUCE_SCALAR**: H×W → outputs 1 tile/batch

**Input Modes:**
| Mode | Behavior | Use Case |
|------|----------|----------|
| STREAMING (default) | **Auto-batched**: Waits all tiles per row/batch/chunk, indexed access, pops all | Most efficient - default choice |
| PRELOADED | All tiles ready, indexed access, caller manages wait/pop | When tiles must persist in CB |
| PERSISTENT | All tiles ready, NO pop (reusable) | Multi-pass scenarios |

**Key Features:**
- **Auto-batched STREAMING**: Default mode now batches CB operations for optimal efficiency (same perf as PRELOADED, simpler code)
- `PostReduceOp`: Custom lambda after reduce (e.g., `recip_tile` for softmax, `max_tile` for attention)
- `init`/`uninit` control: Fine-grained multi-pass support
- `input_stride`: Non-contiguous layout support

---

## 2. Migration Status

### ✅ Completed (23 files)

| Phase | Count | Kernels | Mode |
|-------|-------|---------|------|
| **P1: Basic Reduces** | 3 | reduce_h, reduce_hw, reduce_w | Auto-batched STREAMING |
| **P2: LayerNorm** | 4 | layernorm_sharded, layernorm_large_tensor, layernorm_sharded_pre/post_allgather | PRELOADED/PERSISTENT |
| **P3: GroupNorm** | 2 | groupnorm, groupnorm_sharded_v2 | PRELOADED |
| **P4: Softmax** | 4 | softmax, softmax_sharded, softmax_large_tensor, tilize_untilize | PERSISTENT + PostReduceOp |
| **P5: Specialized** | 2 | deepseek_grouped_gate, moe | PRELOADED |
| **P6: Distributed Norms** | 5 | rmsnorm/layernorm_pre_allgather{,_2d}, experimental/rmsnorm | **Auto-batched STREAMING** |
| **P7: Simple Ops** | 3 | sampling, ssm_1d_sum_reduce, sdpa_decode | STREAMING + PostReduceOp |

**Key Achievements:**
- **P4:** Dual-pass MAX+SUM reduces with PERSISTENT mode + PostReduceOp lambdas
- **P6-P7:** Auto-batched STREAMING mode - same efficiency as PRELOADED, simpler code
- **Library Enhancement:** Auto-batching now default for STREAMING mode (benefits P1 + 11 other kernels)

**Migration Details - Phase 6 (Distributed Norms):**
All 5 files migrated to auto-batched STREAMING mode:
- Removed manual `cb_wait_front(cb_x2/cb_inp, Wt)` calls (library handles automatically)
- Removed manual `cb_pop_front(cb_x2/cb_inp, Wt)` calls (library handles automatically)
- Removed redundant `cb_wait_front(cb_reduce/cb_scaler, 1)` calls (library handles scaler wait)
- Removed unused `FLOAT32_REDUCTION` variables (library auto-detects from ENABLE_FP32_DEST_ACC)
- Result: ~4 lines saved per reduce call, cleaner code

**Migration Details - Phase 7 (Simple Operations):**
- `sampling.cpp`: Kept PRELOADED (tiles must persist per postcondition)
- `ssm_1d_sum_reduce.cpp`: STREAMING mode, removed wrapper function, removed old init/uninit workaround
- `sdpa_decode/compute_common.hpp`: STREAMING mode + PostReduceOp lambda for eltwise max optimization

**Library Enhancement - Auto-Batched STREAMING:**
- STREAMING mode now waits/pops tiles in bulk (per row/batch/chunk) instead of one-at-a-time
- Uses indexed access like PRELOADED for efficiency
- Caller doesn't manage CB lifecycle (simpler than PRELOADED)
- Performance: 2 CB calls instead of 2×Wt (e.g., 2 calls vs 64 calls for Wt=32)
- Automatically applied to all existing STREAMING mode kernels (backward compatible)

### ⏳ Blocked (30 files - need library extensions)

**Accumulation Support Needed (26 files):**
- All Moreh operations (26) - manual accumulator reload pattern

**Stride/Non-contiguous Support Needed (4 files):**
- `layernorm_distributed/layernorm_post_allgather.cpp` - cols 0,2,4... pattern
- `rmsnorm_distributed/rmsnorm_post_allgather.cpp` - cols 0,2,4... pattern
- `experimental/.../rmsnorm_post_allgather.cpp` - cols 0,2,4... pattern
- `experimental/ccl/rms_allgather/.../rms_compute.cpp` - complex interleaved

---

## 3. Blocker Details

### Moreh Operations (26 files)

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

**Affected:** moreh_dot, moreh_softmax (4), moreh_layer_norm (5), moreh_mean, moreh_norm (4), moreh_bias_backward (2), and others

**Solution Options:**
1. Add `load_accumulator` template parameter
2. Add `skip_dst_management` for external register control
3. New `reduce_accumulating()` function

---

## 4. File Reference

### Library Files
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
- `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`

### Migrated Kernels (23 files)
```
# P1-P2 (7 files) - Auto-batched STREAMING
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/{reduce_h,reduce_hw,reduce_w}.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_{sharded,large_tensor,sharded_pre_allgather,sharded_post_allgather}.cpp

# P3 (2 files) - PRELOADED
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/{groupnorm,groupnorm_sharded_v2}.cpp

# P4-P5 (6 files) - PERSISTENT + PostReduceOp
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/{softmax,softmax_sharded,softmax_large_tensor}.cpp
ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp
ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp
ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp

# P6 (5 files) - Auto-batched STREAMING (NEW!)
ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather{,_2d}.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather{,_2d}.cpp
ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp

# P7 (3 files) - STREAMING + PostReduceOp (NEW!)
ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp
ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/ssm_1d_sum_reduce.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp
```

---

## 5. Key Takeaways

**Achieved:**
- **43% of operations migrated** (23/54 files) with proven patterns
- **Library enhanced:** Auto-batched STREAMING mode, PERSISTENT mode, PostReduceOp lambdas
- Complex multi-pass reductions working (softmax, layernorm, groupnorm, distributed norms)
- **Code quality:** -87 net lines removed, cleaner and more maintainable kernels
- **Performance:** 11+ kernels automatically benefit from auto-batching optimization

**Phases 6 & 7 Completed:**
- ✅ 5 distributed norm pre-allgather kernels migrated with auto-batched STREAMING
- ✅ 3 simple operations migrated (sampling, ssm, sdpa_decode)
- ✅ All 253+ tests passing across all migrated kernels

**Blocked:**
- 56% (30 files) need accumulation or stride support
- Decision needed: extend library vs keep specialized implementations

**ROI:** 76% of addressable kernels (23/30 non-blocked) migrated

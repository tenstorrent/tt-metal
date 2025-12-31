# Reduce Helpers Library Migration Plan

## Status Summary (as of 2025-12-30)

**Scope:** `ttnn/cpp/ttnn/operations/` folder only (54 kernel files use `reduce_tile`)

**Progress:**
- ✅ **14/54 migrated (26%)** - Phases 1-5 complete
- 🎯 **8/54 ready (15%)** - No library changes needed
- ⏳ **31/54 blocked (57%)** - Need library extensions
- 🔍 **1/54 TBD (2%)** - Needs investigation

**Next:** Migrate 5 distributed norm pre-allgather kernels (Phase 6)

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
| STREAMING | `cb_wait_front`/`cb_pop_front` per tile | Standard dataflow |
| PRELOADED | All tiles ready, indexed access | Sharded kernels |
| PERSISTENT | All tiles ready, NO pop (reusable) | Multi-pass scenarios |

**Advanced Features:**
- `PostReduceOp`: Custom lambda after reduce (e.g., `recip_tile` for softmax)
- `init`/`uninit` control: Fine-grained multi-pass support
- `input_stride`: Non-contiguous layout support

---

## 2. Migration Status

### ✅ Completed (14 files)

| Phase | Count | Kernels |
|-------|-------|---------|
| **P1: Basic Reduces** | 3 | reduce_h, reduce_hw, reduce_w |
| **P2: LayerNorm** | 4 | layernorm_sharded, layernorm_large_tensor, layernorm_sharded_pre/post_allgather |
| **P3: GroupNorm** | 2 | groupnorm, groupnorm_sharded_v2 |
| **P4: Softmax** | 3 | softmax, softmax_sharded, tilize_untilize |
| **P5: Specialized** | 2 | deepseek_grouped_gate, moe |

**Key Achievement (P4):** Dual-pass MAX+SUM reduces with PERSISTENT mode + PostReduceOp lambdas

### 🎯 Ready for Migration (8 files)

**Phase 6: Distributed Norms (5 files)**
- `rmsnorm_distributed/rmsnorm_pre_allgather.cpp` - single REDUCE_ROW
- `rmsnorm_distributed/rmsnorm_pre_allgather_2d.cpp` - single REDUCE_ROW
- `layernorm_distributed/layernorm_pre_allgather.cpp` - dual REDUCE_ROW
- `layernorm_distributed/layernorm_pre_allgather_2d.cpp` - dual REDUCE_ROW
- `experimental/.../rmsnorm_pre_allgather.cpp` - single REDUCE_ROW

**Phase 7: Simple Operations (3 files)**
- `reduction/sampling/.../sampling.cpp` - PRELOADED pattern
- `experimental/ssm/hc_sum_reduce/.../ssm_1d_sum_reduce.cpp` - single-tile REDUCE_COL
- `transformer/sdpa_decode/.../compute_common.hpp` - indexed pattern

### ⏳ Blocked (31 files - need library extensions)

**Accumulation Support Needed (27 files):**
- `softmax_large_tensor.cpp` (1) - multi-pass with `use_prev_reduce` flag
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

### Migrated Kernels (14 files)
```
# P1-P2 (7 files)
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/{reduce_h,reduce_hw,reduce_w}.cpp
ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_{sharded,large_tensor,sharded_pre_allgather,sharded_post_allgather}.cpp

# P3 (2 files)
ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/{groupnorm,groupnorm_sharded_v2}.cpp

# P4-P5 (5 files)
ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/{softmax,softmax_sharded}.cpp
ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp
ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp
ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp
```

### Next Candidates (8 files)
```
# Phase 6 (5 files)
ttnn/cpp/ttnn/operations/normalization/{rmsnorm,layernorm}_distributed/device/kernels/compute/{rmsnorm,layernorm}_pre_allgather{,_2d}.cpp
ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp

# Phase 7 (3 files)
ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp
ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/ssm_1d_sum_reduce.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp
```

---

## 5. Key Takeaways

**Achieved:**
- 26% of operations migrated with proven patterns
- Library enhanced: PERSISTENT mode + PostReduceOp
- Complex multi-pass reductions working (softmax, layernorm, groupnorm)

**Next Steps:**
- Phase 6: 5 distributed norm files (straightforward PRELOADED patterns)
- Phase 7: 3 simple operations (minimal effort)
- **Result:** 41% coverage with zero library changes

**Blocked:**
- 57% need accumulation or stride support
- Decision needed: extend library vs keep specialized implementations

**ROI:** 64% of addressable kernels already migrated

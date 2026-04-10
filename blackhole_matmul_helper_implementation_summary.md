# Matmul Helper Library Migration Summary (Blackhole)

**Branch:** `wransom/matmulop` | **System:** Blackhole P100A

---

## Overview

Migrated the `MatmulOp<IsBlockMode>` class from `tt_metal/hw/inc/api/compute/matmul_op.h` into a free-function helper library at `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.{hpp,inl}`, following the `compute_kernel_lib` conventions established by the `pjosipovic/llk_helper_library` branch (reduce_helpers, binary_op_helpers, etc.). All production kernel call sites, test kernels, tt-train kernels, and model kernels have been migrated. The original `matmul_op.h` header is no longer included by any kernel.

---

## Files Created

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp` | Declarations, config structs, enums, documentation, usage examples |
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl` | All function implementations (included at bottom of .hpp) |

## API Design

**Namespace:** `compute_kernel_lib` (consistent with existing helpers)

**Key types:**
- `MatmulMode` enum: `TILE` or `BLOCK` -- template parameter for compile-time LLK dispatch
- `MatmulConfig` struct -- replaces `ckernel::MatmulOpConfig`, with `::tile()` and `::block()` factory methods
- `MatmulBlockShape` struct -- dimensions for the fully automated `matmul()` function
- `MoeDm1State` struct -- state for MoE W2 accumulate helpers (moved from `ckernel` namespace)

**6-layer function hierarchy:**

| Layer | Key Functions | Replaces |
|-------|--------------|----------|
| 0 - Init | `matmul_init`, `matmul_init_short`, `*_with_dt`, `*_with_both_dt` | `mm.init()`, `mm.init_short()`, etc. |
| 1 - Single op | `matmul_tile` | `mm.matmul_one_tile()` |
| 2 - Accumulate | `matmul_accumulate`, `*_subblock`, `*_no_mop` | `mm.accumulate()`, `mm.accumulate_tile_subblock()`, `mm.accumulate_no_mop()` |
| 3 - Pack/reload | `matmul_pack_to_cb`, `*_to_partials`, `matmul_reload_partials` | `mm.end_to_output()`, `mm.end_to_partials()`, `mm.reload_partials()` |
| 4 - Compound | `matmul_accumulate_and_pack`, `matmul_compute_one_tile`, `matmul_compute_inner_block` | `mm.accumulate_and_pack()`, `mm.compute_one_tile()`, `mm.compute_inner_block()` |
| 5 - Specialized | `matmul_reduce_w`, `*_with_init`, `*_attn`, `*_reduce_subblock_inplace`, `matmul_moe_*` | All specialized MatmulOp methods |
| 6 - Automated | `matmul(cfg, shape)` | `mm.run()` |

All functions are `ALWI` (always-inline) and templated on `<MatmulMode mode>`.

---

## Files Modified (~28 production kernels + supporting files)

**Core matmul kernels:**
- `ttnn/.../matmul/device/kernels/compute/bmm.cpp`
- `ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm.cpp`
- `ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
- `ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`

**SDPA:**
- `ttnn/.../transformer/sdpa/device/kernels/compute/compute_streaming.hpp`
- `ttnn/.../transformer/sdpa/device/kernels/compute/compute_common.hpp`

**Attention:**
- `ttnn/.../experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp`
- `ttnn/.../experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp`

**Reduction/Moreh:**
- `ttnn/.../reduction/generic/device/kernels/compute/reduce_w.cpp`
- `ttnn/.../moreh/moreh_matmul/device/kernels/moreh_matmul.cpp`
- `ttnn/.../moreh/moreh_mean/device/kernels/moreh_mean_w.cpp`
- `ttnn/.../moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp`

**Conv:**
- `ttnn/.../conv/conv2d/device/kernels/conv_bmm_tilize.cpp`
- `ttnn/.../experimental/conv3d/device/kernels/compute.cpp`

**Deepseek/MoE:**
- `ttnn/.../experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp`
- `ttnn/.../experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp`
- `ttnn/.../experimental/ccl/moe_compute/device/kernels/compute.cpp`
- `ttnn/.../experimental/ccl/moe_gpt/device/kernels/compute.cpp`

**CCL/Minimal:**
- `ttnn/.../experimental/minimal_matmul/device/kernels/compute.cpp`
- `ttnn/.../experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp`
- `ttnn/.../experimental/ccl/llama_all_gather_matmul_async/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp`

**Other:**
- `ttnn/.../experimental/topk_router_gpt/device/kernels/compute.cpp`
- `ttnn/cpp/ttnn/kernel/compute/bmm_tilize_untilize.cpp`

**tt-train:**
- `tt-train/.../sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt-train/.../sdpa_fw/device/kernels/compute/sdpa_compute_utils.hpp`
- `tt-train/.../sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp`

**Models:**
- `models/demos/deepseek_v3_b1/unified_kernels/rope.hpp`

---

## Migration Pattern (for the Wormhole instance to follow)

Each kernel migration follows this pattern:

```cpp
// BEFORE:
#include "api/compute/matmul_op.h"
ckernel::MatmulOpConfig cfg{};
cfg.in0_cb_id = cb_in0; cfg.in1_cb_id = cb_in1; cfg.out_cb_id = cb_out;
cfg.ct_dim = W; cfg.rt_dim = H; cfg.kt_dim = K; cfg.transpose = trans;
ckernel::BlockMatmulOp mm(cfg);
mm.init();
mm.begin_subblock();
mm.reload_partials(n);
mm.accumulate(a, b, 0, K, 1, stride, 0);

// AFTER:
#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp"
using namespace compute_kernel_lib;
auto cfg = MatmulConfig::block(cb_in0, cb_in1, cb_out, W, H, K, trans);
matmul_init<BLOCK>(cfg);
tile_regs_acquire();
matmul_reload_partials<BLOCK>(cfg, n);
matmul_accumulate<BLOCK>(cfg, a, b, 0, K, 1, stride, 0);
```

Key: `mm.begin_subblock()` was just `tile_regs_acquire()` internally. All explicit DEST management (`tile_regs_commit`, `tile_regs_wait`, `tile_regs_release`, `pack_tile`, `cb_reserve_back`, `cb_push_back`) remains unchanged in the kernel files.

---

## Bug Fixed During Migration

**ALWI macro ordering:** The `.hpp` forward declarations used `ALWI` before it was defined by `common_globals.h` (which only comes in through the `.inl`). When `matmul_helpers_compute.hpp` is the first include in a kernel file, this caused compilation failure. Fixed with:

```cpp
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif
```

at the top of the `.hpp`. This is a Blackhole-and-Wormhole concern -- the Wormhole instance should verify this fix is present.

---

## Architecture-Specific Considerations for Merge

1. **`matmul_accumulate_no_mop`** -- Guarded by `#ifdef ARCH_BLACKHOLE`. Calls `ckernel::matmul_block_no_mop()` which only exists on Blackhole. The SDPA streaming kernel (`compute_streaming.hpp`) uses this in an `#ifdef ARCH_BLACKHOLE` / `#else` branch. The Wormhole instance should verify that the `#else` path (`matmul_accumulate<BLOCK>`) is correct on their side.

2. **`matmul_block_math_dynamic_throttle`** -- Blackhole-only firmware-controlled throttle in `matmul.h`. Called internally by `matmul_block()`. No helper-level concern, but worth noting the underlying LLK behavior differs between architectures.

3. **Quasar guards** -- The underlying `matmul.h` has `#ifndef ARCH_QUASAR` guards on many functions. The helpers call through to the same functions so Quasar behavior is inherited.

4. **`mm_block_init_short_with_both_dt`** -- Block mode only, used by `conv_bmm_tilize.cpp` and `transformer_group_attn_matmul.cpp`. The `static_assert(mode == MatmulMode::BLOCK)` enforces this at compile time.

5. **CB count limits** -- 32 on Wormhole, 64 on Blackhole. No helper-level concern (CB IDs are just passed through), but the Wormhole instance should verify none of the migrated kernels accidentally use higher CB indices.

---

## Test Results (Blackhole P100A)

| Suite | Passed | Skipped | Failed |
|-------|--------|---------|--------|
| test_matmul.py (main unit) | 557 | 136 | **0** |
| test_linear + addmm + experimental + batch_mismatch | 482 | 35 | **0** |
| test_sparse_matmul | 11 | 0 | **0** |
| test_conv2d | 161 | 48 | **0** |
| nightly matmul (11 test files) | 1,147 | 994 | **0** |
| moreh_matmul (nightly) | 89 | 84 | **0** |
| SDPA prefill | 3 | 2 | **0** |
| SDPA decode | 9 | 0 | **0** |
| moreh_mean + moreh_sum | 303 | 227 | **0** |
| **TOTAL** | **2,762** | **1,526** | **0** |

All skips are pre-existing (BH-specific limitations, config exclusions, grid size requirements). No test was skipped due to our changes.

---

## What Remains

1. **Wormhole validation** -- The other instance should run equivalent tests on Wormhole. The `#ifdef ARCH_BLACKHOLE` paths in `compute_streaming.hpp` (no-mop) are the most architecture-sensitive.
2. **The old `matmul_op.h` header still exists** -- It's no longer included by any kernel but hasn't been deleted. The master instance should decide whether to remove it or keep it for backwards compatibility.
3. **`.matmul_op_project/` directory** -- Contains old migration examples and design docs from the PoC phase. Can be cleaned up.
4. **MoE/CCL kernels** -- The MoE kernels (`moe_compute`, `moe_gpt`, `topk_router_gpt`) were migrated but I couldn't run their specific tests (they require multi-device setups like T3000/Galaxy). The compilation succeeded and the matmul helper calls are structurally identical, but device-level validation on multi-device setups is needed.

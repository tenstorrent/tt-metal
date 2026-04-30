# Eltwise Helper Phase 2 Verification Report

**Date**: 2026-04-30
**Scope**: 12 high-value claims from Phase 1 investigation files
**Status**: All 12 claims **CONFIRMED**

## Summary

| CLAIM_ID | Verdict | Evidence (file:line) | Brief |
|---|---|---|---|
| rsqrt_template_mismatch | CONFIRMED | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rsqrt.h:13-24` | Init 2 params, exec 4 — mismatch real |
| mac_missing_primitive | CONFIRMED | grep: 0 matches in tt_metal/hw/ckernels | No LLK function exists |
| mask_hardcoded_slot_plus_one | CONFIRMED | `compute/eltwise_unary/mask.h:42`; `ckernel_sfpu_mask.h:20,32,45` | API `idst2_mask` ignored; hardcoded `dst_reg[32]` |
| binary_max_min_share_init | CONFIRMED | `compute/binary_max_min.h:96,180` | Distinct LLK init wrappers; share underlying logic — caller dedup |
| mul_tiles_bcast_not_separate_function | CONFIRMED | `compute/eltwise_binary.h:71-87`; grep 0 matches | Broadcast = template mode, not separate function |
| ternary_slot_order | CONFIRMED | `llk_math_eltwise_ternary_sfpu_{addcmul,where,lerp}.h:13-14` | `(in0, in1, in2, out)` strict order |
| dropout_rand_share_rng | CONFIRMED | `ckernel_sfpu_dropout.h:22-23`; `ckernel_sfpu_rand.h:14-15` | Both program global RNG; mutex required |
| sfpu_params_wrapper_canonical | CONFIRMED | `tt-llk/.../llk_math_eltwise_unary_sfpu_params.h:13-73` | Universal acquire/iterate-faces/release wrapper |
| copy_tile_one_dest_slot | CONFIRMED | `compute/tile_move_copy.h:99-109` | One slot per call; fan-out = N copies |
| dest_to_srcb_reconfig_separate_path | CONFIRMED | `compute/eltwise_binary.h:236-252,278-295` | DEST_TO_SRCA vs DEST_TO_SRCB — distinct unpack paths |
| dest_auto_limit_constexpr | CONFIRMED | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88-102` | `constexpr DEST_AUTO_LIMIT = get_dest_limit()` |
| cumulative_wait_unsupported | CONFIRMED | grep eltwise compute kernels | No growing-window `cb_wait_front(cb, base+i)` pattern |

## Detailed Verdicts

### 1. rsqrt_template_mismatch — CONFIRMED
- Init `template <bool APPROXIMATE, bool legacy_compat>` (line 13)
- Exec `template <bool APPROXIMATE, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat>` (line 18)
- Mismatch real. Helper must route fp32_dest_acc_en + FAST_APPROX through exec only, OR add to init.

### 2. mac_missing_primitive — CONFIRMED
- Zero grep matches for `llk_math_eltwise_ternary_sfpu_mac*` across `tt_metal/hw/ckernels/`.
- "mac" in catalog is unsupported. Flag as gap, exclude from helper.

### 3. mask_hardcoded_slot_plus_one — CONFIRMED
- `compute/eltwise_unary/mask.h:42` — API declares `idst2_mask` parameter.
- `ckernel_sfpu_mask.h:20,32,45` — LLK reads from `dst_reg[32]` unconditionally regardless of API param.
- Encode `Slot+1` at compile time in `Mask<DF, DataSlot>` struct with `static_assert(DataSlot < 7)`.

### 4. binary_max_min_share_init — CONFIRMED
- `compute/binary_max_min.h:96` — `binary_max_tile_init()` calls `llk_math_eltwise_binary_sfpu_binary_max_init()`.
- Line 180 — `binary_min_tile_init()` calls `..._binary_min_init()`.
- Distinct LLK names but same effective init work. Caller dedups when both used.

### 5. mul_tiles_bcast_not_separate_function — CONFIRMED
- `compute/eltwise_binary.h:71-87` — `binary_tiles_init<full_init, EltwiseBinaryType>()`.
- No `BroadcastType` template param at the binary_tiles_init level; broadcast is encoded via separate FPU init pathway with BroadcastType template.
- No separate `mul_tiles_bcast` LLK function.

### 6. ternary_slot_order — CONFIRMED
- All ternary SFPU LLK functions:  `llk_math_eltwise_ternary_sfpu_<op>(uint dst_index0, uint dst_index1, uint dst_index2, uint odst, ...)`.
- Strict slot order (in0, in1, in2, out). No permutation.

### 7. dropout_rand_share_rng — CONFIRMED
- `ckernel_sfpu_dropout.h:22-23` — `dropout_init(const uint seed)` calls `_init_dropout_(seed)`.
- `ckernel_sfpu_rand.h:14-15` — `rand_init(uint32_t seed)` calls `init_prng_seed(seed)`.
- Both program global RNG state; second call overwrites first.

### 8. sfpu_params_wrapper_canonical — CONFIRMED
- `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h:13-73`.
- Acquires DEST, iterates faces (R=2, C=2, RC=4), calls forwarded SFPU func, releases DEST.
- Universal wrapper used by all unary SFPU exec wrappers.

### 9. copy_tile_one_dest_slot — CONFIRMED
- `compute/tile_move_copy.h:99-109` — `copy_tile(in_cb_id, in_tile_index, dst_tile_index)`.
- One DEST slot per call. Fan-out across N slots requires N calls.

### 10. dest_to_srcb_reconfig_separate_path — CONFIRMED
- `compute/eltwise_binary.h:236-252,278-295`.
- `binary_dest_reuse_tiles_init` / `binary_dest_reuse_tiles` templated on `EltwiseBinaryReuseDestType` (DEST_TO_SRCA vs DEST_TO_SRCB).
- Different unpack MOPs per ReuseType. Test independently.

### 11. dest_auto_limit_constexpr — CONFIRMED
- `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88-102`.
- `constexpr uint32_t DEST_AUTO_LIMIT = get_dest_limit();`
- `get_dest_limit()` returns 8/16 for SyncFull or 4/8 for SyncHalf depending on `DST_ACCUM_MODE`.
- Helper must use this, never literal 8.

### 12. cumulative_wait_unsupported — CONFIRMED
- grep across `ttnn/cpp/ttnn/operations/eltwise/*` shows only fixed-count `cb_wait_front(cb, N)` patterns.
- No growing-window `cb_wait_front(cb, base + i)` in production kernels.
- Helper does not need a CumulativeWait policy.

## Design Impact

All 12 design hypotheses confirmed:

1. rsqrt — helper documents/resolves template mismatch
2. mac — out of scope, missing primitive
3. mask — hardcode `Slot+1` at compile time
4. binary_max/min — caller-side dedup; document opportunity
5. mul_tiles_bcast — broadcast is template mode, not separate fn
6. ternary — strict (in0,in1,in2,out) slot order
7. dropout/rand — mutual exclusion required
8. sfpu_params wrapper — canonical, universal
9. copy_tile — single slot, fan-out = N copies (lessons §3.5)
10. DEST reuse — srcA vs srcB are distinct paths, separate tests
11. DEST_AUTO_LIMIT — use constexpr, not literal 8
12. CumulativeWait — not needed in V1 helper

---

Generated: 2026-04-30
Verification Complete

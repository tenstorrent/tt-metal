// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// finalize_sfpu_scope EXPERIMENT — scoped/fused implementations of rms_norm's
// `finalize` post_reduce_op:  1/rms = rsqrt(Sum(x^2) * (1/W) + eps).
//
// ---------------------------------------------------------------------------
// RAW-LLK JUSTIFICATION (what is bypassed, and why the helper cannot express it)
// ---------------------------------------------------------------------------
// Bypassed helpers: `mul_unary_tile` / `add_unary_tile` (binop_with_scalar.h:34-70)
// and `rsqrt_tile` (rsqrt.h:36-45).  All three HARDCODE `VectorMode::RC` and
// `ITERATIONS = 8` at their own call site, so each is a walk over the WHOLE 32x32
// DEST tile = 32 32-lane SFPU vector ops, 96 in total per finalized tile.
//
// The tile they walk is a REDUCE_ROW result: the only meaningful data is COLUMN 0
// (one value per row).  Its single consumer is a `BroadcastDim::Col` operand
// (`rms_col`), which reads column 0 and replicates it across the row; nothing
// downstream reads any other lane.  Two mechanisms recover the waste, neither
// reachable through the wrappers:
//
//   1. SCOPE.  Column 0 lives in faces 0 and 2 (`VectorMode::C`) and, within a
//      face, only in the EVEN column parity — the SFPU walks a face as
//      [rg0-even, rg0-odd, rg1-even, ...], so parity is the INNER axis and
//      `ITERATIONS` (which truncates the OUTER axis) cannot isolate it.  A
//      hand-addressed sfpi body that strides DEST by 2 keeps 8 of the 32 vectors.
//   2. FUSION.  mul, add and rsqrt are three separate WALKS over the same tile.
//      One body that computes `rsqrt(v*inv_w + eps)` per vector makes it one.
//
// Both go through `_llk_math_eltwise_unary_sfpu_params_`, the same entry point
// `SFPU_UNARY_CALL` (and therefore every wrapper above) uses; the only thing
// bypassed is the fixed VectorMode/ITERATIONS the wrapper bakes in.
//
// PRECISION: `APPROX` is never flipped and no approximate rsqrt is used — the
// body calls the identical non-approximate `_calculate_sqrt_body_<APPROX,
// RECIPROCAL=true>` that `rsqrt_tile` calls.  The one numerical difference is
// that the two intermediates stay in fp32 SFPU lane registers instead of round-
// tripping through a 16-bit DEST, i.e. MORE precision, not less.
//
// Also: ONE init instead of two.  `rsqrt_tile_init()` expands to
// `llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>(rsqrt_init)`, which already
// runs `_llk_math_eltwise_unary_sfpu_init_` (SFPU config reg + ADDR_MOD_7/6 +
// counter reset) — exactly what `binop_with_scalar_tile_init()`
// (`SFPU_UNARY_INIT(unused)` -> the same invariant + a counter reset) exists to
// do.  With the binop gone, its init is redundant.

#pragma once

#include <cstdint>
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

// Variant ids (-DRMS_FINALIZE_VARIANT):
//   0  stock chain (in the compute kernel; not here)
//   1  fused body + even-parity stride        -- 8 vectors, 1 init   (CANDIDATE)
//   2  fused body + even-parity stride, negative-input NaN guard dropped
//   3  fused body, VectorMode::RC             -- 32 vectors (fusion without scope)
#define RMS_FIN_FUSED_CSKIP 1
#define RMS_FIN_FUSED_CSKIP_NONNEG 2
#define RMS_FIN_FUSED_RC 3

#ifdef TRISC_MATH
// NOT unrolled, deliberately: the sqrt body alone nearly fills the 8 SFPU LREGs,
// and holding the two scalar constants live across an UNROLLED copy of it spills
// ("internal compiler error: cannot store sfpu register" -- measured).  One body
// live at a time; measured at the same ns/vector as the stock unrolled body.
//
// NONNEG passes FAST_APPROX to the sqrt body.  On the NON-approximate branch
// (which is what runs: APPROX is fixed by the user's math_approx_mode) that flag
// does exactly one thing -- it elides the trailing `v_if(x < 0) y = NaN` guard.
// It does not change the Newton iteration or any value for x >= 0, and the
// finalize's input is a sum of squares times 1/W plus a positive epsilon, so
// x < 0 is unreachable.
template <int STRIDE, int NITER, bool NONNEG>
sfpi_inline void rms_finalize_body(uint32_t inv_w_bits, uint32_t eps_bits) {
#pragma GCC unroll 1
    for (int i = 0; i < NITER; i++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat v = sfpi::dst_reg[0] * inv_w + eps;
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, NONNEG>(v);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
#endif

template <uint32_t VARIANT>
ALWI void rms_finalize_scoped(uint32_t dst_idx, uint32_t inv_w_bits, uint32_t eps_bits) {
    rsqrt_tile_init();
    if constexpr (VARIANT == RMS_FIN_FUSED_CSKIP) {
        // 4 even-parity vectors per face x faces {0, 2} = 8 vectors, covering
        // column 0 of all 32 rows.  Net dst_reg advance is 4*2 == the stock
        // ITERATIONS 8, so VectorMode::C's face0 -> face2 stepping composes.
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            rms_finalize_body<2, 4, false>, dst_idx, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
    } else if constexpr (VARIANT == RMS_FIN_FUSED_CSKIP_NONNEG) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            rms_finalize_body<2, 4, true>, dst_idx, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
    } else {  // RMS_FIN_FUSED_RC -- fusion without the scope (whole tile stays valid)
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            rms_finalize_body<1, 8, false>, dst_idx, ckernel::VectorMode::RC, inv_w_bits, eps_bits)));
    }
}

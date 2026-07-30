// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// SiTU, the two halves of Moonshot's SituAndMul activation:
//   softcap(x) = beta * tanh(x / beta)                 -- the up half
//   situ_gate(x)  = beta * tanh(x / beta) * sigmoid(x)    -- the gate half
// The full activation is situ_gate(gate) * softcap(up), the product being an
// ordinary elementwise multiply outside these ops. Kimi K3 uses beta 4 for the
// gate half and 25 for the up half, but nothing here is specific to those.
//
// beta arrives as an fp32 bit pattern with 1/beta precomputed by the caller, so
// the kernel never divides (same contract as celu's alpha / alpha_recip).
//
// Non-finite inputs: +/-Inf clamps to +/-beta, which is what keeps a stale Inf
// from escaping. NaN does NOT propagate -- min(., 1.0) below compiles to SFPSWAP
// MIN, whose magnitude compare picks 1.0 over NaN, so the result is finite
// (+beta, or 0 for situ_gate at bf16 dst). Stock calculate_tanh behaves the same.
//
// One init serves both ops: tanh owns all three vConstFloatPrgm registers and the
// sigmoid half borrows none of them (see _situ_reciprocal_). Any OTHER SFPU op in
// the same kernel that owns those registers -- silu, sigmoid, exp, reciprocal --
// must be re-inited before its next use.
inline void situ_init() { tanh_init</*APPROXIMATION_MODE=*/false, /*is_fp32_dest_acc_en=*/false>(); }

// Newton reciprocal, structurally identical to sfpu_reciprocal_iter but carrying
// its 2.0 as a literal. That function reads the constant from vConstFloatPrgm0,
// which situ_init has loaded with a Sollya tanh coefficient -- borrowing it here
// would silently corrupt both halves. Keep the two in sync when recip.h changes.
template <int MAX_ITER>
sfpi_inline sfpi::vFloat _situ_reciprocal_(const sfpi::vFloat x) {
    // SFPARECIP: a 7-bit-mantissa seed from a 128-entry LUT indexed by the top 7
    // mantissa bits, so 0.9944/x < seed < 1.0054/x. One Newton step below takes
    // that to ~14 bits (past bf16's 8), two to ~28 (past fp32's 24) -- which is
    // where MAX_ITER's 1-vs-2 comes from.
    //
    // Outside 2**-126 <= |x| < 2**126 the instruction returns a saturated value
    // rather than an approximation: +inf for |x| < 2**-126 (ALL subnormals, not
    // just zero) and 0 for |x| >= 2**126 (~8.5e37, note that is below FLT_MAX).
    // Both are already exactly 1/x, and the guard below leaves them unrefined.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Normally t = 2.0 - x * y, but we negate it (and negate again via y = y * -t
    // below). On Blackhole x=0 with y=inf (and vice versa) gives t=+NaN regardless
    // of operand signs; negating the meaning of t turns NaN detection into a
    // trivial sign check, since every comparison against NaN is false and the
    // degenerate cases then keep the already-correct seed. v_if (t >= 2.0) on the
    // un-negated form would be equivalent, but SFPI has no SFPLE/SFPGT.
    //
    // The trailing `- 0.0f` on each y update is instruction shape, not arithmetic:
    // SFPMAD is a fused multiply-ADD with no bare-multiply form. Subtracting zero
    // rather than adding it also preserves a signed zero result.
    sfpi::vFloat t = x * y - 2.0f;
    if constexpr (MAX_ITER > 1) {
        sfpi::vFloat y1 = y * -t - 0.0f;
        // If t=NaN then t>=0. This check consumes the SFPNOP slot of the preceding
        // SFPMAD, so the predicate is free here.
        v_if(t < 0) {
            t = x * y1 - 2.0f;
            y = y1 * -t - 0.0f;
        }
        v_endif;
    } else {
        // If t=NaN then t>=0. Unlike the two-iteration form above this check cannot
        // hide in an SFPNOP slot -- it depends on the immediately preceding SFPMAD.
        v_if(t < 0) { y = y * -t - 0.0f; }
        v_endif;
    }
    return y;
}

// sigmoid(x) = 1 / (1 + exp(-x)). Both exp variants are free of vConstFloatPrgm,
// so this composes with tanh under a single init.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _situ_sigmoid_(sfpi::vFloat x) {
    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_dest_acc_en) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }
    return _situ_reciprocal_<is_fp32_dest_acc_en ? 2 : 1>(1.0f + exp_neg_x);
}

// beta * tanh(x / beta), unrounded.
//
// Always the Sollya polynomial, never _sfpu_tanh_fp32_accurate_, in BOTH dst
// modes and for BOTH ops. The SFPU has no spill path, so exceeding the LReg file
// is a hard compile abort ("cannot store sfpu register"), not a slowdown. Two
// independent things each exhaust it:
//
//   * the accurate expm1 tanh together with the gate half's sigmoid;
//   * the accurate expm1 tanh together with runtime beta, on its own. Measured:
//     softcap alone at fp32 dst compiles with beta as a constexpr and aborts with
//     beta and 1/beta as live vFloats. The two pinned registers are the whole
//     margin, so making this dst-mode-dependent is not available while beta is a
//     runtime parameter -- which it must be for a public ttnn op.
//
// Reproducing either needs -O3 WITHOUT -flto; under LTO codegen is deferred and
// the abort moves to link time or disappears.
//
// The polynomial costs ~2.3e-3 relative, below a bf16 pack and below the ~4e-3 a
// cheaper sigmoid exp would have cost instead. Callers needing fp32-grade tanh
// should use tanh_tile.
sfpi_inline sfpi::vFloat _situ_softcap_(sfpi::vFloat x, sfpi::vFloat beta, sfpi::vFloat inv_beta) {
    return _sfpu_tanh_polynomial_(x * inv_beta) * beta;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_softcap(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_beta = Converter::as_float(param1);

    for (int d = 0; d < ITERATIONS; d++) {
        // Round once, after the beta rescale -- rounding tanh first would discard
        // bits that the multiply by beta then amplifies.
        sfpi::vFloat result = _situ_softcap_(sfpi::dst_reg[0], beta, inv_beta);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// The gate half. sigmoid takes the RAW x, not the capped value.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_situ_gate(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_beta = Converter::as_float(param1);

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat result = _situ_softcap_(x, beta, inv_beta) * _situ_sigmoid_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

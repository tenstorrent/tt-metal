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

// Newton reciprocal carrying its 2.0 as a literal. sfpu_reciprocal_iter reads that
// constant from vConstFloatPrgm0, which situ_init has loaded with a Sollya tanh
// coefficient -- borrowing it here would silently corrupt both halves.
template <int MAX_ITER>
sfpi_inline sfpi::vFloat _situ_reciprocal_(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);

    // t carries the negation of (2 - x*y) so the NaN produced on Blackhole when x
    // and y are 0/inf is caught by a plain sign check.
    sfpi::vFloat t = x * y - 2.0f;
    if constexpr (MAX_ITER > 1) {
        sfpi::vFloat y1 = y * -t - 0.0f;
        v_if(t < 0) {
            t = x * y1 - 2.0f;
            y = y1 * -t - 0.0f;
        }
        v_endif;
    } else {
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
// modes. The accurate expm1 form leaves too little LReg headroom to coexist with
// the gate half's sigmoid: the pair overruns the register file and the SFPU has no
// spill path, so the compiler aborts with "cannot store sfpu register". Dropping
// to the polynomial costs ~2.3e-3 relative, below both a bf16 pack and the ~4e-3
// the alternative (a cheaper sigmoid exp) would have cost. Callers needing
// fp32-grade tanh should use tanh_tile.
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

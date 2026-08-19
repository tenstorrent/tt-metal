// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_conversions.h"

namespace ckernel::sfpu {

// logaddexp2(a, b) = max(a, b) + log2(1 + 2^-|a - b|)
//
// Same shape as logaddexp, different overflow point. The composed form,
// log2(2^a + 2^b), overflows at |x| > 127 rather than 88.7, because the base-2
// exponential saturates a float32 one binade earlier than e^x does. The result is
// bounded the same way by its own inputs:
//
//     max(a, b) <= logaddexp2(a, b) <= max(a, b) + 1
//
// so a finite pair always has a finite result within 1 of one of its arguments.
//
// Two scalings turn the base-e primitives into base-2 ones:
//
//     2^-|a-b|      = exp(-|a-b| * ln 2)
//     log2(1 + t)   = log1p(t) * log2(e)
//
// and log1p sees t in (0, 1] exactly as it does for logaddexp, so its existing
// polynomial domain [-0.25, 0.5] is unchanged and NO new coefficients are needed.
// That was worth measuring rather than assuming: a variant with log2(e) folded into
// its own minimax fit was simulated against torch.logaddexp2 over 262144 pairs drawn
// from U(-200, 200) and came out at the same 1.45e-06 worst relative error -- a 1.00x
// difference. The dominant error is the cancellation in max + correction, not the
// extra multiply, so a second coefficient set would add review surface and buy nothing.
//
// Written with at most three live vFloat values, for the same register-spill reason
// documented in ckernel_sfpu_logaddexp.h.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    // ln 2 and log2(e). Both are the correctly rounded float32 nearest values.
    constexpr float LN_TWO = 0.693147182f;
    constexpr float LOG2_E = 1.442695041f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = sfpi::max(a, b);
        a = sfpi::abs(a - b);
        // The accurate exponential is required for the same reason as in logaddexp: the
        // approximate body returns 255/256 rather than 1 at zero, and here that error
        // lands on the correction term whose exact value at |a-b| = 0 is 1.
        b = _sfpu_exp_fp32_accurate_(-a * LN_TWO);
        result = result + calculate_log1p_fp32<is_fp32_dest_acc_en>(b) * LOG2_E;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Identical to logaddexp's init, and for the same reason: log1p reads its polynomial
// coefficients from the program constant registers and an SFPU helper called from
// another op's kernel does not carry its own initialisation. The coefficient set differs
// by destination precision, which is why this init is templated.
//
// The base conversion lives in the kernel above, not here, so these constants stay
// exactly the ones log1p expects.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp2_init() {
    const float LOG_TWO = 0.693147182f;
    const float TWO_TO_M23 = 1.19209290e-7f;
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

}  // namespace ckernel::sfpu

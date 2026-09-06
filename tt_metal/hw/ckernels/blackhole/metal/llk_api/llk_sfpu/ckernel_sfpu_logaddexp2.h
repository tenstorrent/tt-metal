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
// The composed form, log2(2^a + 2^b), overflows at |x| > 127 (2^128 is not
// representable) and underflows below -149, even though the result is bounded by its
// own inputs: max(a,b) <= logaddexp2(a,b) <= max(a,b) + 1. Here the argument of the
// power is -|a - b| <= 0, so it lands in (0, 1] and cannot overflow; the magnitude comes
// from max(a, b), which is representable by assumption.
//
// Evaluated through the base-e primitives that ckernel_sfpu_logaddexp.h already relies on:
//     2^-|a - b|    = exp(-|a - b| * ln 2)
//     log2(1 + t)   = log1p(t) * log2(e)
// No new polynomial: log1p sees exactly the same (0, 1] argument range as it does in
// logaddexp, so its existing coefficient set applies unchanged. A variant with log2(e)
// folded into its own fit was simulated against torch.logaddexp2 over 262144 pairs from
// U(-200, 200) and was indistinguishable from this form at every percentile measured; the
// dominant error is the cancellation in max + correction, not the extra multiply.
//
// Written with at most three live vFloat values, for the same register-spill reason
// documented in ckernel_sfpu_logaddexp.h.
//
// Equal infinities need their own branch, exactly as in logaddexp: |a - b| is the right
// difference everywhere except a == b == +/-inf, where inf - inf is NaN and the NaN then
// swallows the whole result. SFPU float equality does not reliably match either infinity
// sign on device, so infinity is classified from its exponent/mantissa fields and the two
// operands are required to have identical signed bit patterns. Substituting a zero
// difference then keeps both signs correct: max(+/-inf, +/-inf) + 1 = +/-inf. The added
// clause excludes NaNs, so they do not take this equal-infinity fix-up and still propagate.
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
        v_if(sfpi::exexp(a) == 128 && sfpi::exman(a) == 0 && sfpi::as<sfpi::vInt>(a) == sfpi::as<sfpi::vInt>(b)) {
            a = 0.0f;
        }
        v_else { a = sfpi::abs(a - b); }
        v_endif;
        // The accurate exponential is required for the same reason as in logaddexp: the
        // approximate body returns 255/256 rather than 1 at zero, and here that error
        // lands on the correction term whose exact value at |a - b| = 0 is 1.
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

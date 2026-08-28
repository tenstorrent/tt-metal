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

// logaddexp(a, b) = max(a, b) + log1p(exp(-|a - b|))
//
// The composed form, log(exp(a) + exp(b)), overflows at |x| > 88.7 even though the
// result is bounded by its own inputs: max(a,b) <= logaddexp(a,b) <= max(a,b) + ln 2.
// Here the exponential argument is -|a - b| <= 0, so exp() lands in (0, 1] and cannot
// overflow; the magnitude comes from max(a, b), which is representable by assumption.
//
// Written with at most three live vFloat values. The straightforward version, holding
// the two inputs plus max, difference, exponential and result at once, does not fit:
// the SFPI compiler reports "cannot store sfpu register (register spill)".
//
// Equal infinities need their own branch. |a - b| is the right difference everywhere
// except a == b == +/-inf, where inf - inf is NaN and the NaN then swallows the whole
// result; the composed form this replaces returns +/-inf there, so without the branch
// the fix would be a regression on those two points. Comparing first and substituting a
// zero difference keeps it correct: max(inf, inf) + ln 2 = inf. A NaN input still
// propagates, because a NaN compares unequal to itself and takes the other branch.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = sfpi::max(a, b);
        v_if(a == b) { a = 0.0f; }
        v_else { a = sfpi::abs(a - b); }
        v_endif;
        // The accurate exponential is required, not a preference: the approximate body
        // returns 255/256 rather than 1 at zero, which lands as a 2.8e-03 relative error
        // on the whole result. _sfpu_exp_fp32_accurate_unsafe_ is also not usable here --
        // it drops the underflow guard, and -|a-b| reaches large negative values.
        b = _sfpu_exp_fp32_accurate_(-a);
        result = result + calculate_log1p_fp32<is_fp32_dest_acc_en>(b);

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// log1p reads its polynomial coefficients from the program constant registers, so they
// have to be loaded here: an SFPU helper called from another op's kernel does not carry
// its own initialisation. Without this, calculate_log1p_fp32 returns 2^24 instead of ln 2.
//
// The coefficient set differs by destination precision, which is why this init is
// templated where the surrounding binary inits are not.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp_init() {
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

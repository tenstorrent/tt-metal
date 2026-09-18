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
// the fix would be a regression on those two points. SFPU float equality does not
// reliably match either infinity sign on device, so classify infinity from its
// exponent/mantissa fields and require identical signed bit patterns.
// Substituting a zero difference then keeps both signs correct:
// max(+/-inf, +/-inf) + ln 2 = +/-inf. The added clause excludes NaNs, so they do
// not take this equal-infinity fix-up.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = sfpi::max(a, b);
        v_if(sfpi::exexp(a) == 128 && sfpi::exman(a) == 0 && sfpi::as<sfpi::vInt>(a) == sfpi::as<sfpi::vInt>(b)) {
            a = 0.0f;
        }
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
    // Delegating to log1p_init rather than copying its values keeps one source for the
    // tuned coefficients: a retune of the log1p polynomial reaches this op instead of
    // silently desyncing from it. log1p_init ignores its first two template parameters.
    log1p_init<false /* APPROXIMATION_MODE */, false /* FAST_APPROX */, is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu

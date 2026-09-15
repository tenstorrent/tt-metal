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

// Special-value handling shared by logaddexp and logaddexp2, which are both evaluated as
//     max(a, b) + correction(|a - b|)
// Call _sfpu_logaddexp_max_ before _sfpu_logaddexp_gap_: the gap relies on a NaN pair
// already having been copied into the result.

// max(a, b), with a NaN in either operand propagated rather than left to the arithmetic.
// max() is a bare SFPSWAP with no NaN guard, so it orders a NaN by its sign: a negative NaN
// loses to any finite operand and would drop out of the result. Copying the NaN operand
// into the result settles it, because SFPMAD addition is IEEE-754 for a non-finite input:
// NaN + correction stays NaN whatever the correction makes of the NaN difference.
sfpi_inline sfpi::vFloat _sfpu_logaddexp_max_(const sfpi::vFloat& a, const sfpi::vFloat& b) {
    sfpi::vFloat result = sfpi::max(a, b);
    v_if(sfpi::exexp(a) == 128 && sfpi::exman(a) != 0) { result = a; }
    v_elseif(sfpi::exexp(b) == 128 && sfpi::exman(b) != 0) { result = b; }
    v_endif;
    return result;
}

// Replaces a with |a - b|, or with zero when the operands are bit-identical. Equal
// infinities are why the clause exists: inf - inf is NaN, and that NaN would swallow a
// result the composed form gets right, while a zero gap keeps both signs correct because
// max(+/-inf, +/-inf) plus a finite correction is +/-inf. The comparison goes through
// as<vInt>, so the clause is bit identity by construction rather than depending on how
// vFloat equality lowers. For any other bit-identical pair the substitution changes
// nothing: a finite difference is already +0.0, and a NaN pair was already copied into
// the result.
sfpi_inline void _sfpu_logaddexp_gap_(sfpi::vFloat& a, const sfpi::vFloat& b) {
    v_if(sfpi::as<sfpi::vInt>(a) == sfpi::as<sfpi::vInt>(b)) { a = 0.0f; }
    v_else { a = sfpi::abs(a - b); }
    v_endif;
}

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
// Equal infinities and NaN operands are handled by the two helpers above, which
// logaddexp2 shares. Without the gap helper the fused form would regress on equal
// infinities, which the composed form returns as +/-inf.
//
// APPROXIMATION_MODE is accepted and ignored, as in calculate_log1p_fp32: the exponential
// below is always the accurate one, because the approximate body is not accurate enough here.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_logaddexp_max_(a, b);
        _sfpu_logaddexp_gap_(a, b);
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

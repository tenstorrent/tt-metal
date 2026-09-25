// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel::sfpu {

// Shared by logaddexp and logaddexp2, which are both evaluated as
//     max(a, b) + correction(|a - b|)
//
// Leaves max(a, b) in result and returns the gap |a - b|, with every special case folded
// into one test on the gap. The gap is NaN exactly when an operand is NaN or the operands
// are equal infinities (inf - inf). For those lanes a + b is already the answer, NaN or
// the shared infinity, so it replaces max(a, b), whose SFPSWAP is a bare swap that orders
// a NaN by its sign. The gap becomes 0 so that the correction added afterwards is finite
// (ln 2, or 1 in base 2) and cannot disturb that answer. Opposite infinities give an
// infinite, not NaN, gap and a zero correction, so max(a, b) already holds the right
// infinity.
sfpi_inline sfpi::vFloat _sfpu_logaddexp_max_gap_(const sfpi::vFloat& a, const sfpi::vFloat& b, sfpi::vFloat& result) {
    result = sfpi::max(a, b);
    sfpi::vFloat gap = sfpi::abs(a - b);
    v_if(sfpi::is_nan(gap)) {
        result = a + b;
        gap = 0.0f;
    }
    v_endif;
    return gap;
}

// log1p(y) for y in [0, 1], the only range a logaddexp correction produces, for a bfloat16
// destination:
//     log1p(y) ~= y + y^2 * (c0 + c1 * y + c2 * y^2)
// with c0..c2 in the three program constant registers (calculate_sfpu_logaddexp_init), so
// no coefficient is reloaded per iteration. The linear term is kept exact, which keeps the
// relative error small as y -> 0, and the fit is constrained so that y = 1, the equal-input
// case, returns ln 2 correctly rounded to fp32. Maximum relative error on [0, 1] is 2^-10.8,
// below the 2^-8.8 of the general bfloat16 log1p, which has to cover (-1, inf) and whose
// NaN and infinity checks can never fire on this range.
sfpi_inline sfpi::vFloat _sfpu_logaddexp_log1p_unit_bf16_(const sfpi::vFloat& y) {
    sfpi::vFloat r = sfpi::vConstFloatPrgm2 * y + sfpi::vConstFloatPrgm1;
    r = r * y + sfpi::vConstFloatPrgm0;
    return r * (y * y) + y;
}

// logaddexp(a, b) = max(a, b) + log1p(exp(-|a - b|))
//
// The composed form, log(exp(a) + exp(b)), overflows at |x| > 88.7 even though the
// result is bounded by its own inputs: max(a,b) <= logaddexp(a,b) <= max(a,b) + ln 2.
// Here the exponential argument is -|a - b| <= 0, so exp() lands in (0, 1] and cannot
// overflow; the magnitude comes from max(a, b), which is representable by assumption.
//
// The two inputs are dead once max(a, b) and the gap are known, and the exponential and
// the correction reuse their registers. The straightforward version, holding the two
// inputs plus max, difference, exponential and result at once, does not fit: the SFPI
// compiler reports "cannot store sfpu register (register spill)".
//
// APPROXIMATION_MODE is accepted and ignored, as in log1p_init: the exponential below is
// chosen by the destination precision instead, and is never the approximate body, which is
// not accurate enough here.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result;
        a = _sfpu_logaddexp_max_gap_(a, b, result);

        if constexpr (is_fp32_dest_acc_en) {
            // Guarded against the large negative values -|a-b| reaches, which the unguarded
            // _sfpu_exp_fp32_accurate_unsafe_ is not.
            b = _sfpu_exp_fp32_accurate_(-a);
            result = result + calculate_log1p_fp32<true>(b);
        } else {
            // exp_21f, the exponential exp itself uses for a bfloat16 result. Its argument
            // is -|a-b| <= 0, so the only bound it needs is the lower one: capping the gap at
            // 88 keeps val / ln 2 + 127 in [0, 256), the range the unclamped body requires,
            // at the same point where the clamped body would have saturated to 2^-127. The
            // body rounds its result to bfloat16, which the correction depends on: exp_21f
            // alone returns 1.0017 at zero, and the rounding makes it exactly 1, so equal
            // inputs get ln 2. The approximate body is not usable either: it returns
            // 255/256 rather than 1 at zero.
            b = _sfpu_exp_21f_bf16_unsafe_<false>(-sfpi::min(a, 88.0f));
            result = result + _sfpu_logaddexp_log1p_unit_bf16_(b);

            // SFPSTORE would truncate to bfloat16, so round first, the way calculate_log1p and
            // exp round their own bfloat16 results. The hardware rounds a tie away from zero
            // rather than to even, which can only matter when the fp32 sum lands exactly
            // halfway between two bfloat16 values.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        // ADDR_MOD_6 (calculate_sfpu_logaddexp_init) advances the destination on the store,
        // in place of a separate dst_reg++.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode(ADDR_MOD_6) = result;
    }
}

// The corrections read their polynomial coefficients from the program constant registers,
// so they have to be loaded here: an SFPU helper called from another op's kernel does not
// carry its own initialisation. The same goes for ADDR_MOD_6, which the kernels store
// through.
//
// The coefficient set differs by destination precision, which is why this init is
// templated where the surrounding binary inits are not.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp_init() {
    // The store advances the destination by one SFPU row (dest.incr = 2, what dst_reg++
    // does), so each iteration saves a TTINCRWC. The binary SFPU init only sets ADDR_MOD_6
    // this way for binary max/min and the comparisons, not for this op, so it is set here.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);

    if constexpr (is_fp32_dest_acc_en) {
        // Delegating to log1p_init rather than copying its values keeps one source for the
        // tuned coefficients: a retune of the log1p polynomial reaches this op instead of
        // silently desyncing from it. log1p_init ignores its first two template parameters.
        // Without this, calculate_log1p_fp32 returns 2^24 instead of ln 2.
        log1p_init<false /* APPROXIMATION_MODE */, false /* FAST_APPROX */, true>();
    } else {
        // c0, c1, c2 of _sfpu_logaddexp_log1p_unit_bf16_.
        sfpi::vConstFloatPrgm0 = -0x1.f48448p-2f;
        sfpi::vConstFloatPrgm1 = 0x1.074d5ap-2f;
        sfpi::vConstFloatPrgm2 = -0x1.3402cep-4f;
    }
}

}  // namespace ckernel::sfpu

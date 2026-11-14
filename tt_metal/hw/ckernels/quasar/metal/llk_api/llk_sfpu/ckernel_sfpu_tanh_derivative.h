// SPDX-FileCopyrightText: ? 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_polyval.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Copied from the Blackhole metal ckernel_sfpu_tanh_derivative.h: only the sech^2 kernel that
// tanh_derivative_tile dispatches. The legacy 1 - lut(x)^2 kernel and its SFPLUTFP32 table are
// not ported: Quasar has no LReg-operand sfpi::lut, and nothing calls that kernel.

constexpr float SECH2_FLUSH_LIMIT = 45.0f;

// Cody-Waite split of -ln2, and the degree-6 minimax of _sfpu_exp_fp32_accurate_ for exp(r).
constexpr float SECH2_INV_LN2 = 1.4426950408889634f;
constexpr float SECH2_LN2_HI = -0.6931152343750000f;  // -ln2, high 13 bits
constexpr float SECH2_LN2_LO = -3.19461832987e-05f;   // -ln2 - SECH2_LN2_HI
constexpr float SECH2_C2 = 4.99999851e-1f, SECH2_C3 = 1.66664720e-1f, SECH2_C4 = 4.16695364e-2f;
constexpr float SECH2_C5 = 8.37312452e-3f, SECH2_C6 = 1.37805939e-3f;

// 4·exp(-2a) to fp32 accuracy for 0 <= a < SECH2_FLUSH_LIMIT.
//
// Cody-Waite reduction -2a = k·ln2 + r, the minimax above for exp(r) on |r| <= ln2/2, and
// 2^(k+2) written straight into the exponent field. k·LN2_HI is exact (a 13-bit constant
// times |k| <= 130) and so is its sum with -2a (Sterbenz), so r carries only the LN2_LO
// rounding. The ×4 goes into the exponent *before* the FTZ test, so the result stays normal
// out to a ≈ 44.4 where exp(-2a) alone is already subnormal; folding it into the argument as
// +ln4 instead would cost ~60 fp32 ULP at the far end.
//
// Every fp32 constant here costs two SFPLOADI per row if written as a literal, and the
// compiler does not hoist them. So the reduction constants come in as vFloats the caller
// loads once before its row loop, and C2..C4 sit in vConstFloatPrgm0..2 (programmed by
// tanh_derivative_sech2_init). C5 stays a literal: a fourth hoisted vFloat runs out of LRegs.
sfpi_inline sfpi::vFloat sech2_exp4_neg2x(
    sfpi::vFloat a, sfpi::vFloat inv_ln2, sfpi::vFloat ln2_hi, sfpi::vFloat ln2_lo) {
    sfpi::vFloat t = a * -2.0f;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(t * inv_ln2, k_int);
    sfpi::vFloat r = k * ln2_hi + t;
    r = k * ln2_lo + r;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, SECH2_C5, SECH2_C6);

    sfpi::vInt e = sfpi::exexp(p, sfpi::ExponentMode::Biased) + k_int + 2;
    sfpi::vFloat result = 0.0f;
    v_if(e > 0) { result = sfpi::setexp(p, e); }  // FTZ
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_tanh_derivative_sech2() {
    // Loop-invariant: loaded once here instead of two SFPLOADI each per row. Non-const on
    // purpose; a const vFloat live across the row loop fails to compile.
    sfpi::vFloat inv_ln2 = SECH2_INV_LN2;
    sfpi::vFloat ln2_hi = SECH2_LN2_HI;
    sfpi::vFloat ln2_lo = SECH2_LN2_LO;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        // sech² is even. Clear the sign bit rather than take sfpi::abs, which leaves a
        // sign-set NaN sign-set and let it through the guard (-NaN gave 1.8e-38 with an fp32
        // destination and +inf with a bf16 one; torch writes every bf16 NaN as 0xFFFF).
        sfpi::vFloat a = sfpi::setsgn(val, 0);

        v_if(a < SECH2_FLUSH_LIMIT) {
            sfpi::vFloat e4 = sech2_exp4_neg2x(a, inv_ln2, ln2_hi, ln2_lo);  // 4e, exact at x = 0 (the whole chain is)

            // den = (1 + e)² = e4·q + 1 with q = (e + 2)/4, a single fused rounding. That
            // rounding and q's are recovered exactly -- 0.5 - q by Sterbenz, 1 - den because
            // den ∈ [1, 4] lies on a grid no finer than 1's -- and carried to the end as den_lo.
            sfpi::vFloat q = e4 * 0.0625f + 0.5f;
            sfpi::vFloat q_lo = e4 * 0.0625f + (0.5f - q);
            sfpi::vFloat den = e4 * q + 1.0f;
            sfpi::vFloat den_lo = e4 * q + (1.0f - den);
            den_lo = e4 * q_lo + den_lo;

            // 1/den, two Newton steps from the SFPARECIP seed as in _sfpu_reciprocal_gt0_<true>.
            // den ∈ [1, 4] with both ends attained (x = 0 gives 4, a flushed e4 gives 1), so
            // the seed needs no zero/inf/NaN guard.
            sfpi::vFloat y = sfpi::approx_recip(den);
            y = y * (-den * y + 1.0f) + y;
            y = y * (-den * y + 1.0f) + y;

            // Round the quotient once: q0 = e4·y, then the residual e4 - q0·(den + den_lo)
            // put back through y. What is left at 2 ULP is the exp's own rounding; carrying its
            // low part as well gets 1 ULP everywhere but costs ~256 cycles/tile more.
            sfpi::vFloat q0 = e4 * y;
            sfpi::vFloat w = -q0 * den + e4;
            w = -q0 * den_lo + w;
            result = w * y + q0;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // SFPSTORE truncates; round to bfloat16 explicitly.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_sech2_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // The exp polynomial's C2..C4, read by sech2_exp4_neg2x. This takes all three programmable
    // constants, so sfpu_reciprocal_iter (which wants Prgm0 = 2.0) cannot be swapped in for the
    // SFPARECIP-seeded reciprocal without moving one of them back to a literal.
    sfpi::vConstFloatPrgm0 = SECH2_C2;
    sfpi::vConstFloatPrgm1 = SECH2_C3;
    sfpi::vConstFloatPrgm2 = SECH2_C4;
}

}  // namespace sfpu
}  // namespace ckernel

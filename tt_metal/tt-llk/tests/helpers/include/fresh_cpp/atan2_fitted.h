// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane DH, 2026-08-20).
// Fitted atan2 built from the tt-polynomial-fitter frontier's EXISTING unary
// atan winner — the fit is TAKEN from the frontier, not refit:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/atan_p8_s1_uniform_basis_ulp.csv (BH, bf16;
//                  pareto_winners.csv selected=1 row for atan: basis P8/s1)
//   kernel shape : tt-metal branch nkapre/tt-polynomial-fitter @ 8063ae8eced6529bd5fa9d8336066601eaa4fd67
//                  generic_lut_activation kernels, piecewise_generic basis path
//                  (BASIS_SIGNED_ABS_POLY, emit signed_abs, NO clamp — atan.json
//                  basis block has no clamp): y = copysign(P(|x|), x) with the
//                  EXPANDED P(u) = u*Q(u) (c0 == 0 exactly), one segment, plain
//                  Horner — the arithmetic order the frontier receipts measured.
//   Recorded claim (silicon BH/BF16 frontier, summary_bf16.csv): unary atan
//   max_ulp_pure_bf16 0.49925696849823 at 1.71 us vs TTNN 128.0 ulp @ 3.80 us
//   (superior BOTH axes; fit expr atan(x)/x, emit domain [0, pi/2]).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//
// COMPOSITION (this lane — the frontier has no atan2; the fitter's own CR-lane
// coverage table records "binary op; fitter list has unary atan only"): the
// standard atan2 quadrant fixup of fresh_cpp/atan2.h wraps the frontier fit:
//   t = min(|y|,|x|) / max(|y|,|x|)  in [0, 1]  ⊂  the fit's emit domain
//   [0, pi/2], so the polynomial is applied strictly inside its fitted range.
//   The measured kernel's |x| / copysign(., x) are absorbed by the reduction:
//   t >= 0 by construction, and the sign returns through the pre-composed
//   quadrant fold (angle = B + F*P(t)) + copysign(., y).  atan2(0, 0) = 0.
//   Only the polynomial (and its evaluation order) is fitter-measured; the
//   divide + quadrant arithmetic is the same storm-S1 octant fixup as the
//   fresh row's arm and is judged by the row's TRUE-atan2 golden.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive the atan selection from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv and
//   refresh the coefficient block + shas here.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_atan2_fitted_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    // atan_p8_s1_uniform_basis_ulp.csv segment 0 (c0 = 0 exactly; expanded
    // signed-abs basis: P(u) = u*Q(u), coefficients c1..c8 verbatim).
    constexpr float C1   = 1.0000040531158447e+00f;
    constexpr float C2   = -4.3038243893533945e-04f;
    constexpr float C3   = -3.2570558786392212e-01f;
    constexpr float C4   = -5.1757059991359711e-02f;
    constexpr float C5   = 3.7205523252487183e-01f;
    constexpr float C6   = -3.0121427774429321e-01f;
    constexpr float C7   = 1.0751627385616302e-01f;
    constexpr float C8   = -1.5071129426360130e-02f;
    constexpr float PI   = 3.14159265358979323846f;
    constexpr float PI_2 = 1.57079632679489661923f;

#pragma GCC unroll 0
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 0
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat y  = sfpi::dst_reg[0];
            const sfpi::vFloat x  = sfpi::dst_reg[tile_rows];
            const sfpi::vFloat ay = sfpi::abs(y);
            const sfpi::vFloat ax = sfpi::abs(x);

            // Quadrant folds pre-composed into one affine map (B + F*p) before
            // the polynomial, exactly as in fresh_cpp/atan2.h (keeps the peak
            // SFPU register pressure inside the 8-LREG file):
            //   octant fold      p -> pi/2 - p   ((B,F) = (pi/2, -1))
            //   left half-plane  p -> pi   - p   (B -> pi - B, F -> -F)
            sfpi::vFloat fold_base  = 0.0f;
            sfpi::vFloat fold_scale = 1.0f;
            v_if (ay > ax)
            {
                fold_base  = PI_2;
                fold_scale = -1.0f;
            }
            v_endif;
            v_if (x < 0.0f)
            {
                fold_base  = PI - fold_base;
                fold_scale = -fold_scale;
            }
            v_endif;

            const sfpi::vFloat hi = sfpi::max(ay, ax);
            sfpi::vFloat t        = sfpi::min(ay, ax) * fresh_recip_positive(hi);
            // Both operands zero: the angle of the origin is defined as 0
            // (the fold above left (B,F) = (0,1) on those lanes).
            v_if (hi == 0.0f)
            {
                t = 0.0f;
            }
            v_endif;

            // Frontier fit: plain Horner over t (t in [0,1], t >= 0 stands in
            // for the measured kernel's |x|), expanded basis c0 == 0.
            sfpi::vFloat p = C8;
            p              = p * t + C7;
            p              = p * t + C6;
            p              = p * t + C5;
            p              = p * t + C4;
            p              = p * t + C3;
            p              = p * t + C2;
            p              = p * t + C1;
            p              = p * t; // + c0 == 0

            // The folded angle lies in [0, pi]; the result takes y's sign.
            sfpi::vFloat angle = sfpi::copysgn(fold_scale * p + fold_base, y);
            if constexpr (!DST_ACCUM_MODE)
            {
                angle = sfpi::convert<sfpi::vFloat16b>(angle, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = angle;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_atan2_fitted_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fitted atan2 expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fitted atan2 expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fitted atan2 expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_atan2_fitted_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu

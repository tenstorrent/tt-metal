// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `atan2` corpus row (metal
// calculate_sfpu_atan2).  Mathematical definition (torch.atan2): the
// quadrant-aware arctangent of y/x with y = in0, x = in1.  Stated by the
// standard octant reduction plus a published arctangent minimax polynomial:
//
//   t = min(|y|,|x|) / max(|y|,|x|)        (t in [0, 1])
//   p = arctan(t)                          (Abramowitz & Stegun 4.4.49
//                                           degree-9 odd minimax, |eps| <= 1e-5)
//   if |y| >  |x|:  p = pi/2 - p           (octant fold)
//   if  x  <  0:    p = pi   - p           (left half-plane)
//   result = copysign(p, y);  atan2(0, 0) = 0.
//
// The match is under the suite's tolerance + PCC gate (the golden notes the
// production kernel is itself a minimax approximation).
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_atan2_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    // Abramowitz & Stegun 4.4.49 coefficients (odd powers of t).
    constexpr float A1   = 0.9998660f;
    constexpr float A3   = -0.3302995f;
    constexpr float A5   = 0.1801410f;
    constexpr float A7   = -0.0851330f;
    constexpr float A9   = 0.0208351f;
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

            // Fold both reductions into one affine map BEFORE the polynomial
            // (angle = B + F*p), so ay/ax/x go dead early and the peak SFPU
            // register pressure stays inside the 8-LREG file:
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

            const sfpi::vFloat t2 = t * t;
            const sfpi::vFloat p  = ((((A9 * t2 + A7) * t2 + A5) * t2 + A3) * t2 + A1) * t;

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
inline void call_atan2_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh atan2 expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh atan2 expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh atan2 expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_atan2_fresh_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu

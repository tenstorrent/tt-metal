// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// div_int32_floor — LEGACY body, preserved UNWIRED (laneEK 2026-08-21;
// the castfp32tofp16a_legacy.h / lcm_legacy.h precedent).  Superseded by
// the fresh_recip_hwseed rewrite in div_int32_floor.h; kept as the
// historical statement (storm contract,
// fresh_cpp/README.md).  Floor division over the node's positive Int32
// domain ([1, 8e6] both operands, test_sfpu_binary._INT_BINARY_STIMULI),
// where floor == trunc and every operand converts to fp32 exactly
// (< 2^23):
//   q0 = round_nearest(a * (1/b)), then at most +/-2 integer fixups on the
//   exact residual r = a - q*b (integers < 2^24 make the fp32 product and
//   subtraction exact) until 0 <= r < b.
// 1/b is the magic-constant Newton reciprocal (three refinements, ~1 ulp:
// fresh_cpp/digamma.h fresh_recip_positive_blinn, the header of record); the
// round-nearest integer split uses the 2^23 rounding-bias identity (valid
// for 0 <= q < 2^23, which the domain guarantees).  Exactness validated
// over 2e5 random domain samples: laneS2-evidence-20260819/fit_s2.py.
// Golden: torch.div(rounding_mode="floor")
// (golden_generators._div_int32_floor), exact Int32.  Production: metal
// ckernel_sfpu_div_int32_floor.h.
#include <cstdint>

#include "digamma.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_div_int32_floor_fresh_cpp_legacy()
{
    constexpr std::uint32_t tile_rows = 32;
    constexpr float BIAS              = 8388608.0f; // 2^23

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();

            const sfpi::vFloat fa = sfpi::int32_to_float(a, sfpi::RoundMode::Nearest); // exact: |a| < 2^23
            const sfpi::vFloat fb = sfpi::int32_to_float(b, sfpi::RoundMode::Nearest);

            // Nearest-integer quotient estimate from the refined reciprocal.
            const sfpi::vFloat q_est = fa * fresh_recip_positive_blinn(fb);
            const sfpi::vFloat t     = q_est + BIAS;
            sfpi::vInt qi            = sfpi::as<sfpi::vInt>(t) - sfpi::as<sfpi::vInt>(sfpi::vFloat(BIAS));
            sfpi::vFloat qf          = t - BIAS;

            // Exact residual fixups: |q0 - floor(a/b)| <= 2 by construction.
            sfpi::vFloat rem = fa - qf * fb;
            for (int fix = 0; fix < 3; ++fix)
            {
                v_if (rem < 0.0f)
                {
                    qf  = qf - 1.0f;
                    qi  = qi - 1;
                    rem = rem + fb;
                }
                v_endif;
                v_if (rem >= fb)
                {
                    qf  = qf + 1.0f;
                    qi  = qi + 1;
                    rem = rem - fb;
                }
                v_endif;
            }

            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = qi;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_div_int32_floor_fresh_cpp_legacy(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh div_int32_floor expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh div_int32_floor expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh div_int32_floor expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (the fresh add/sub/mul
    // precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_div_int32_floor_fresh_cpp_legacy<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu

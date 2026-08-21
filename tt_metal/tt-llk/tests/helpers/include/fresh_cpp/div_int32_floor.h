// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// div_int32_floor — canonical semantic C++ body (storm contract,
// fresh_cpp/README.md).  Floor division over the node's positive Int32
// domain ([1, 8e6] both operands, test_sfpu_binary._INT_BINARY_STIMULI),
// where floor == trunc and every operand converts to fp32 exactly
// (< 2^23):
//   q0 = round_nearest(a * (1/b)), then ONE +/-1 integer fixup on the
//   exact residual r = a - q0*b.
// 1/b is the hardware-seeded reciprocal (SFPARECIP + two MAD-form Newton
// refinements: fresh_common.h fresh_recip_hwseed, the laneDJ recipe that
// replaced the magic-bit-seed statement in addcdiv/fmod/remainder — 5
// slots at dependency depth 5 versus Blinn+3NR's 13 at depth 11); the
// round-nearest integer split uses the 2^23 rounding-bias identity
// (valid for 0 <= q < 2^23, which the domain guarantees).
//
// Exactness CERTIFIED over the full contract domain, laneDI certificate
// pattern (laneEK-evidence-20260821/divint32floor_cert.c, craq bit-exact
// SFPU FMA model + the pinned simulator's SFPARECIP seed lift, every
// fused/unfused rounding assignment of the MAD sites):
//   - per-divisor margin |q0 - a/b| < 2 for ALL b and every assignment
//     (max 0.830), so |q0 - floor(a/b)| <= 1 everywhere;
//   - hence every residual is the EXACT integer a - q0*b in (-b, 2b)
//     (all magnitudes < 2^24, exact under fused AND unfused rem sites)
//     and the single fixup round below recovers floor exactly;
//   - end-to-end boundary sweep (every quotient boundary +/-2, 2.57e9
//     evaluations) zero failures.
// The legacy Blinn-seeded three-round body is preserved unwired in
// div_int32_floor_legacy.h.
// Golden: torch.div(rounding_mode="floor")
// (golden_generators._div_int32_floor), exact Int32.  Production: metal
// ckernel_sfpu_div_int32_floor.h.
#include <cstdint>

#include "fresh_common.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_div_int32_floor_fresh_cpp()
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

            // Hardware-seeded reciprocal (positive finite divisors only,
            // which the domain guarantees).
            const sfpi::vFloat r = fresh_recip_hwseed(fb);

            // Nearest-integer quotient via the 2^23 bias identity.
            const sfpi::vFloat t  = fa * r + BIAS;
            sfpi::vInt qi         = sfpi::as<sfpi::vInt>(t) - sfpi::as<sfpi::vInt>(sfpi::vFloat(BIAS));
            const sfpi::vFloat qf = t - BIAS;

            // Exact residual, ONE certified fixup round: |q0 - floor| <= 1.
            sfpi::vFloat rem = fa - qf * fb;
            v_if (rem < 0.0f)
            {
                qi  = qi - 1;
                rem = rem + fb;
            }
            v_endif;
            v_if (rem >= fb)
            {
                qi = qi + 1;
            }
            v_endif;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = qi;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_div_int32_floor_fresh_cpp(
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
    calculate_div_int32_floor_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu

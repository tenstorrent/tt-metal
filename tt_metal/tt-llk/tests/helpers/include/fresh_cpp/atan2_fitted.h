// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — LICENSED REFIT (lane GT 2026-08-25), superseding the lane DH
// PLACEHOLDER (frontier unary atan winner P8/s1 TAKEN at 0.499 ulp + exact
// Blinn reciprocal — measured +109.14% vs hand: the accuracy CONTRACT, not
// the compiler, priced that arm; laneEA DECISION-PACK-fitted-accuracy Group A
// + owner accuracy-license ratification 2026-08-24).
//
// This arm refits the atan core AT THE HAND ARM'S MEASURED ACCURACY with the
// laneCW interval-LP fitter form:
//   fit tool : tt-polynomial-fitter agent/rlibm-refits @ c73c618a633,
//              scripts/rlibm_interval_fit.py lp_interval_fit (margin-max LP,
//              scipy HiGHS) — run as laneGT-evidence-20260825/accuracy-oracle/
//              fit_licensed.py `lp7` over the exact-bf16 + dense a-grid on
//              [0,1] with uniform half-width 0.004 (the licensed abs budget),
//              coefficients quantized to 1-slot SFPLOADI fp16 immediates
//              (+-1-ulp polish).  Achieved margin 0.967; the margin-LP and
//              the atan2 row's plain minimax refit CONVERGE to the same
//              quantized deg-7 coefficient set (recorded, expected: same
//              objective geometry at this loose target).
//   license  : hand bf16 arm measured EXHAUSTIVELY over the row's golden
//              universe (bf16 pairs |y|,|x| <= 5, +-0 + normals) with
//              bit-exact pinned-sim models: max_abs 0.010575243 /
//              max_pure_bf16_ulp 128.66.  This body ties both extrema
//              exactly (shared SFPARECIP/FTZ mechanism) — EQUAL-OR-BETTER.
//              Proof: laneGT-evidence-20260825/accuracy-oracle/atan2_bars.c.
//
// COMPOSITION (unchanged from the storm-S1 fixup, now strict-octant):
//   a = min(|y|,|x|) * approx_recip(max(|y|,|x|))  in [0,1]
//   r = a + a^3*Q(a^2)  (deg-7 odd, coefficients above)
//   strict |y| > |x|: r = pi/2 - r;  x < 0 (sign bit, incl -0): r = pi - r
//   result = copysgn(r, y);  atan2(0, 0) = 0 via the a = 0 window.
//   Judged by the row's TRUE-atan2 golden under the suite tolerance + PCC.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_atan2_fitted_cpp()
{
#if __riscv_xtttensixwh
    // SFPARECIP is not available on Wormhole (fresh_common.h hwseed
    // discipline): refuse cleanly at instantiation.
    static_assert(fresh_hwseed_supported_on_wh<ITERATIONS>::value, "licensed atan2 refit requires BH/QSR SFPARECIP");
#else
    constexpr std::uint32_t tile_rows = 32;
    // laneGT lp7 margin-LP refit (see provenance header).
    constexpr float C3      = -0.0438537598f; // 0xbd33a000
    constexpr float C2      = 0.155273438f;   // 0x3e1f0000
    constexpr float C1      = -0.326171875f;  // 0xbea70000
    constexpr float HALF_PI = 1.57079632679489661923f;

    // Loop-invariant register residents (L6/L7 are free in the row body):
    // the pi/2 anchor (pi derives from it by an exact exponent add) and the
    // one polynomial coefficient that needs a register (C2/C1 ride as
    // SFPADDI immediates).
    const sfpi::vFloat half_pi = HALF_PI;
    const sfpi::vFloat c3      = C3;

#pragma GCC unroll 0
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 0
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat y  = sfpi::dst_reg[0];
            const sfpi::vFloat x  = sfpi::dst_reg[tile_rows];
            const sfpi::vFloat ax = sfpi::setsgn(x, 0);
            const sfpi::vFloat ay = sfpi::setsgn(y, 0);

            auto [lo, hi] = sfpi::min_max(ax, ay);

            sfpi::vFloat a = lo * sfpi::approx_recip(hi);
            v_if (hi == 0.0f)
            {
                a = 0.0f;
            }
            v_endif;

            const sfpi::vFloat s = a * a;
            sfpi::vFloat q       = c3;
            q                    = q * s + C2;
            q                    = q * s + C1;
            sfpi::vFloat r       = (q * s) * a + a;

            v_if (sfpi::as<sfpi::vInt>(sfpi::setsgn(x, 0)) < sfpi::as<sfpi::vInt>(hi))
            {
                r = half_pi - r;
            }
            v_endif;
            v_if (x < 0.0f)
            {
                r = sfpi::addexp(half_pi, 1) - r;
            }
            v_endif;

            if constexpr (!DST_ACCUM_MODE)
            {
                r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = sfpi::copysgn(r, y);
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
#endif
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

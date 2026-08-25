// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// LICENSED semantic body for the `atan2` corpus row (metal
// calculate_sfpu_atan2), lane GT 2026-08-25, under the owner-ratified
// accuracy license (2026-08-24: hand-matched-or-better error; laneGI
// mechanics).  Mathematical definition (torch.atan2): the quadrant-aware
// arctangent of y/x with y = in0, x = in1, stated by the standard octant
// reduction plus a degree-7 odd minimax arctangent refit AT THE HAND ARM'S
// MEASURED ACCURACY CLASS:
//
//   a = min(|x|,|y|) * approx_recip(max(|x|,|y|))     (a in [0,1];
//       SFPARECIP ~7-bit hardware reciprocal — the SAME reciprocal class the
//       production hand kernel's bf16 arm uses, licensed by its measured bar)
//   r = a + a^3*(C1 + C2*a^2 + C3*a^4)                (deg-7 odd minimax
//       refit of atan, grid minimax LP + fp16-immediate quantization;
//       poly err 1.504e-4 vs the hand poly's own 7.302e-4)
//   if |y| >  |x|:  r = pi/2 - r                      (strict octant fold —
//       strictness makes atan2(+-0, +0) = +-0 exact without a tie window)
//   if  x  <  0:    r = pi   - r                      (incl. x = -0: the
//       sign-bit SETCC gives torch's atan2(+-0, -0) = +-pi)
//   result = copysgn(r, y);  both zero: a forced to 0 (the reciprocal seed
//       is inf at 0 and 0*inf = NaN on the shared FMA; one CC window).
//
// ACCURACY LICENSE PROOF (exhaustive, laneGT-evidence-20260825/
// accuracy-oracle/): over the row's golden input universe (all bf16 pairs
// (y,x) with |y|,|x| <= 5: +-0 and every normal; bf16 denormals are
// unreachable from the fp32 uniform(-5,5) stimuli grid), bit-exact pinned-sim
// instruction models give hand max_abs 0.010575243 / max_pure_bf16_ulp
// 128.66 and THIS body max_abs 0.010575243 / 128.66 — extrema exactly tie
// the hand arm (both are SFPARECIP-error / FTZ-boundary driven, the shared
// mechanism), composite dominance EQUAL-OR-BETTER.  A deg-5 refit measured
// 0.0116 > bar and was REJECTED by the oracle.  The match to the row's
// TRUE-atan2 golden remains under the suite's tolerance + PCC gate.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_atan2_fresh_cpp()
{
#if __riscv_xtttensixwh
    // SFPARECIP is not available on Wormhole (fresh_common.h hwseed
    // discipline): refuse cleanly at instantiation, keep aggregate WH
    // compiles of this header working.
    static_assert(fresh_hwseed_supported_on_wh<ITERATIONS>::value, "licensed atan2 body requires BH/QSR SFPARECIP");
#else
    constexpr std::uint32_t tile_rows = 32;
    // Degree-7 odd minimax refit (laneGT fit_licensed.py remez7: minimax LP
    // over the bf16-and-dense a-grid on [0,1], coefficients quantized to
    // 1-slot SFPLOADI fp16 immediates with +-1-ulp polish).
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

            // a = min/max via the ~7-bit hardware reciprocal estimate
            // (SFPARECIP RECIP) — the hand arm's own reciprocal class,
            // licensed by its measured accuracy bar.
            sfpi::vFloat a = lo * sfpi::approx_recip(hi);
            // Both operands zero: approx_recip(0) = inf and 0*inf = NaN;
            // the angle of the origin is 0 (folds below give +-0 / +-pi).
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

            // Octant fold, STRICT |y| > |x| (integer compare on the
            // nonnegative magnitudes): ties need no fixup (r ~ pi/4 either
            // way) and both-zero lanes stay on the r = 0 path.
            v_if (sfpi::as<sfpi::vInt>(sfpi::setsgn(x, 0)) < sfpi::as<sfpi::vInt>(hi))
            {
                r = half_pi - r;
            }
            v_endif;
            // Left half-plane (sign-bit test: includes x = -0, giving
            // torch's atan2(+-0, -0) = +-pi).
            v_if (x < 0.0f)
            {
                r = sfpi::addexp(half_pi, 1) - r;
            }
            v_endif;

            if constexpr (!DST_ACCUM_MODE)
            {
                r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
            }
            // The folded angle lies in [0, pi]; the result takes y's sign.
            sfpi::dst_reg[0] = sfpi::copysgn(r, y);
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
#endif
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

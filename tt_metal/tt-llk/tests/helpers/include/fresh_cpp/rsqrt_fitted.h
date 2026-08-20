// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted rsqrt vendored from the tt-polynomial-fitter frontier selection:
//   parameters   : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/rsqrt_p2_s3_uniform_any_ulp.csv METADATA
//                  (eval_method newton_root — a STANDALONE method: the CSV's
//                  polynomial segment rows are unused, the algorithm is fully
//                  declared by newton_root_magic 0x5f3759df, newton_root_c1 1.5,
//                  newton_root_reciprocal True, newton_root_iters 2).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — newton_root_rsqrt (also on tt-metal
//                  branch nkapre/tt-polynomial-fitter @ 8063ae8eced6).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P2/s3):
//   max_ulp_pure_bf16 0.4993, 2.10 us vs TTNN 0.4993 ulp @ 2.47 us.
//   This is a DIFFERENT algorithm from the production / fresh rsqrt (the
//   SQRT_23-bits Kokosinski/Moroz seed with the RECIPROCAL refinement arm,
//   fresh_cpp/rsqrt.h): the classic inverse-sqrt magic seed IS 1/sqrt(x)
//   directly, so no final reciprocal composition is needed.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>
#include <limits>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted rsqrt (frontier winner, newton_root standalone): classic
// inverse-sqrt magic seed + two Newton steps y <- y * (1.5 - 0.5*x*y*y).
// Mirrors newton_root_rsqrt arithmetic order (half_x via exponent decrement).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_rsqrt_fitted_cpp()
{
    constexpr int MAGIC        = 0x5f3759df; // newton_root_magic (inverse-sqrt seed)
    constexpr float THREE_HALF = 1.5f;       // newton_root_c1
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x      = sfpi::dst_reg[0];
        sfpi::vFloat y            = sfpi::as<sfpi::vFloat>(sfpi::vInt(MAGIC) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1));
        const sfpi::vFloat half_x = sfpi::addexp(x, -1); // 0.5*x (exponent decrement)
        // newton_root_iters = 2.
        y = y * (THREE_HALF - half_x * (y * y));
        y = y * (THREE_HALF - half_x * (y * y));
        v_if (x < 0.0f)
        {
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if (x == 0.0f)
        {
            y = std::numeric_limits<float>::infinity();
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

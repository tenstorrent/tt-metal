// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the tanhshrink op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the PyTorch reference
// (torch tanhshrink — the production golden):
//
//   tanhshrink(x) = x - tanh(x)
//
// stated per format arm with the production's own minimax fits (they ARE the
// golden math the row's tolerance is fitted to):
//   small |x| <= 1: x - tanh(x) = x^3 * Q(x^2) (cancellation-free residual fit);
//   large |x| >  1: tanh(|x|) directly — fp32 arm by exp/sigmoid identity
//     (Cody-Waite exp + Newton reciprocal), bf16 arm by a cubic on [1, 3.3]
//     saturating at 1 — then x - copysgn(tanh|x|, x).
// The production hand-isms removed: the fp32 arm's vConstFloatPrgm0 = 2.0
// reciprocal parking and the "#pragma GCC unroll 8" pin; every constant is a
// plain local and the loop is free.

#include <cstdint>

// Shared helper (fresh_recip) still lives in the legacy header pending full
// migration (fresh_cpp/README.md legacy note).
#include "fresh_cpp_operations.h"

namespace ckernel::sfpu
{

template <bool IS_FP32_DEST_ACC, int ITERATIONS>
__attribute__((noinline)) void calculate_tanhshrink_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x  = sfpi::dst_reg[0];
        const sfpi::vFloat ax = sfpi::abs(x);
        const sfpi::vFloat u  = x * x;

        // Small-|x| residual arm as the all-lane default: (x - tanh x)/x^3
        // minimax in u = x^2 (production/golden constants per format arm).
        sfpi::vFloat result;
        if constexpr (IS_FP32_DEST_ACC)
        {
            sfpi::vFloat q = 4.2079269770e-04f;
            q              = q * u + -2.5107525289e-03f;
            q              = q * u + 8.2192532718e-03f;
            q              = q * u + -2.1660288796e-02f;
            q              = q * u + 5.3934831172e-02f;
            q              = q * u + -1.3333128393e-01f;
            q              = q * u + 3.3333331347e-01f;
            result         = x * u * q;
        }
        else
        {
            sfpi::vFloat q = -1.0762925260e-02f;
            q              = q * u + 4.8076551408e-02f;
            q              = q * u + -1.3223160803e-01f;
            q              = q * u + 3.3329936862e-01f;
            result         = x * u * q;
        }

        v_if (ax > 1.0f)
        {
            if constexpr (IS_FP32_DEST_ACC)
            {
                // tanh(a) = 2/(1 + exp(-2a)) - 1 on a = min(|x|, 9) (tanh(9)
                // rounds to 1 in fp32).  Cody-Waite exp, degree-5 Horner
                // (production/golden constants), Newton reciprocal.
                const sfpi::vFloat a = sfpi::min(ax, 9.0f);
                const sfpi::vFloat t = -2.0f * a;

                sfpi::vInt k_int;
                const sfpi::vFloat j = fresh_round_nearest(t * 1.442695f, k_int);
                sfpi::vFloat f       = j * -6.93145752e-1f + t;
                f                    = j * -1.42860677e-6f + f;
                sfpi::vFloat r       = 1.37805939e-3f;
                r                    = r * f + 8.37312452e-3f;
                r                    = r * f + 4.16695364e-2f;
                r                    = r * f + 1.66664720e-1f;
                r                    = r * f + 4.99999851e-1f;
                const sfpi::vFloat y = r * f + 1.0f;
                r                    = y * f + 1.0f;
                const sfpi::vFloat e = sfpi::setexp(r, sfpi::exexp(r, sfpi::ExponentMode::Biased) + k_int);

                const sfpi::vFloat sig     = fresh_recip<2>(1.0f + e);
                const sfpi::vFloat tanh_ax = 2.0f * sig - 1.0f;
                // The residual keeps the UNCLAMPED magnitude: for |x| > 9 the
                // saturated tanh subtracts ~1 from the true |x|.
                result = sfpi::copysgn(ax - tanh_ax, x);
            }
            else
            {
                // bf16 arm: cubic minimax of tanh on [1, 3.3], saturating at 1
                // (production/golden constants).
                sfpi::vFloat p = 5.3348409333e-02f;
                p              = p * ax + -4.0859283753e-01f;
                p              = p * ax + 1.0561303143e+00f;
                p              = p * ax + 6.1829000893e-02f;
                p              = sfpi::min(p, 1.0f);
                result         = x - sfpi::copysgn(p, x);
            }
        }
        v_endif;

        if constexpr (!IS_FP32_DEST_ACC)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

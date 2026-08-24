// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// roundingops — LEGACY semantic bodies, preserved verbatim (symbols renamed
// *_legacy) when lane GH rewrote fresh_cpp/roundingops.h's floor (2026-08-24):
// the copysgn(fresh_trunc_magnitude(|v|)) + second-fixup composition below
// compiles to a 23-slot replay row (two chained CC fixup regions, double
// setcc pairs from the '>' spellings) and measured +74.53% vs the hand
// 14-word row (weekly-20260823); the live floor is the ceil-fresh 14-slot
// idiom mirrored.  Kept for A/B archaeology; not wired to any test node.
// The live bodies are fresh_cpp/roundingops.h.
//
// Original header:
// Canonical semantic bodies for the roundingops op file (storm contract,
// fresh_cpp/README.md).  Production: legacy ckernel_sfpu_rounding_ops.h —
// _calculate_floor_/_calculate_trunc_/_calculate_frac_ all bottom out in the
// raw-TTI l_reg-pinned _trunc_body_ (LREG0..3, SFPEXEXP CC gymnastics, mask
// shift); only _calculate_round_ is typed.  The float Ceil variant's semantic
// body predates the storm (calculate_ceil_fresh_cpp, ceil-fresh row); this
// header states the remaining three variants over the shared typed
// truncate-toward-zero primitive (fresh_cpp/helpers.h fresh_trunc_magnitude:
// the 2^23 mantissa-shift round with downward fixup — exact for every finite
// input, pass-through for |v| >= 2^23, inf, NaN).  Goldens: math.floor /
// math.trunc / x - math.trunc(x), non-finite passthrough.
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_trunc_fresh_cpp_legacy()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = sfpi::copysgn(fresh_trunc_magnitude(sfpi::abs(v)), v);
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_floor_fresh_cpp_legacy()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = sfpi::copysgn(fresh_trunc_magnitude(sfpi::abs(v)), v);
        // Truncation rounds toward zero; floor rounds down, so negative
        // non-integers step one below their truncation.
        v_if (r > v)
        {
            r = r - 1.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_frac_fresh_cpp_legacy()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        const sfpi::vFloat t = sfpi::copysgn(fresh_trunc_magnitude(sfpi::abs(v)), v);
        sfpi::vFloat f       = v - t;
        // Non-finite passthrough (the golden's contract): inf - inf would
        // otherwise manufacture a NaN; NaN already propagates through v - t.
        v_if (sfpi::abs(v) == std::numeric_limits<float>::infinity())
        {
            f = v;
        }
        v_endif;
        sfpi::dst_reg[0] = f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

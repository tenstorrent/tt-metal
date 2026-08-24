// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic bodies for the roundingops op file (storm contract,
// fresh_cpp/README.md).  Production: legacy ckernel_sfpu_rounding_ops.h —
// _calculate_floor_/_calculate_trunc_/_calculate_frac_ all bottom out in the
// raw-TTI l_reg-pinned _trunc_body_ (LREG0..3, SFPEXEXP CC gymnastics, mask
// shift); only _calculate_round_ is typed.  The float Ceil variant's semantic
// body predates the storm (calculate_ceil_fresh_cpp, ceil-fresh row); this
// header states the remaining three variants.  Trunc/frac ride the shared
// typed truncate-toward-zero primitive (fresh_cpp/helpers.h
// fresh_trunc_magnitude: the 2^23 mantissa-shift round with downward fixup —
// exact for every finite input, pass-through for |v| >= 2^23, inf, NaN);
// floor is the ceil-fresh idiom mirrored directly (lane GH 2026-08-24, see
// the body comment).  Goldens: math.floor / math.trunc /
// x - math.trunc(x), non-finite passthrough.
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_trunc_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = sfpi::copysgn(fresh_trunc_magnitude(sfpi::abs(v)), v);
        sfpi::dst_reg++;
    }
}

// Floor (lane GH 2026-08-24 rewrite; previous trunc-composition body
// preserved in fresh_cpp/roundingops_legacy.h, unwired).  The exact mirror
// of the ceil-fresh idiom (calculate_ceil_fresh_cpp, fresh_cpp_operations.h):
// round to nearest on the magnitude via the 2^23 mantissa shift, guard
// |v| >= 2^23 / inf / NaN by exponent passthrough, then step DOWN the lanes
// that rounded above the input — nearest is within 1/2 of v, so the
// round-or-round-minus-one pair always lands on floor(v).  One CC fixup
// region instead of the legacy two (trunc's internal downward fixup + the
// floor step), and the fixup comparison is spelled `v < r` so it lowers to
// the single-SFPSETCC canonical form (the legacy `r > v` spellings each
// paid a two-setcc pair): 14 replay slots/row vs the legacy 23 — the
// production hand row's own word count (ceil-fresh class).  Equivalence to
// the legacy body and to floorf proven exhaustively over all 2^32 fp32 bit
// patterns (both SFPU FTZ conventions): laneGH-evidence-20260824/
// floor_equiv_cert.c.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_floor_fresh_cpp()
{
    constexpr float MANTISSA_SHIFT = 8388608.0f; // 2^23
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = v;
        // |v| + 2^23 - 2^23 rounds away the fraction (nearest-even) for all
        // |v| < 2^23; larger magnitudes (and inf/NaN) keep r = v below.
        sfpi::vFloat t = sfpi::abs(v) + MANTISSA_SHIFT;
        t              = t - MANTISSA_SHIFT;
        v_if (sfpi::exexp(v) < 23)
        {
            r = sfpi::copysgn(t, v);
        }
        v_endif;
        // Nearest-integer above the input means r = ceil(v); floor = r - 1.
        v_if (v < r)
        {
            r = r - 1.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_frac_fresh_cpp()
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

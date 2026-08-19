// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the log1p corpus row (storm contract — see
// README.md in this directory).  Include from an LLK_TRISC_MATH kernel after
// sfpu_operations.h; plain typed C++ only.

#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

// Fresh typed-C++ log1p stating the production bf16 golden contract
// (torch.log1p, Float16_b dest, non-approx) independently, one datum per row:
//
//   u = 1 + a = 2^k * t with t in [0.75, 1.5), m = t - 1 in [-0.25, 0.5):
//     log1p(a) = k*ln(2) + log1p(m),
//     log1p(m) ~= m + m^2 * (c0 + m*(c1 + m*c2)).
//
// k is carried in exponent-bit units (k << 23) and recovered with integer
// arithmetic on the IEEE-754 encodings — the published reduction the
// production kernel derives from (Norbert Juffa's log1p; source and license
// in the production header ckernel_sfpu_log1p.h) — but every constant is a
// plain local (production parks ln(2)*2^-23 and two polynomial coefficients
// in vConstFloatPrgm0..2 at init and pins one step on a raw
// __builtin_rvtt_sfpmad), and the 2^-k - 1 term is recovered through a plain
// positive 4.0 anchor instead of the production's negated-anchor MAD fold.
//
// Ordering note: the reconstruction keeps r = ((c2*m + c1)*m + c0)*(m*m) + m
// with the Horner chain multiplied by m before m^2.  At the u == 0 boundary
// (a == -1) the reduction produces a huge-magnitude negative m whose square
// overflows to +inf; the odd power of m in the chain is what signs the
// overflow to the golden -inf, so this grouping is golden math, not a
// scheduling choice.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_log1p_fresh_cpp()
{
    // ln(2) * 2^-23: the final k*ln(2) term consumes k in exponent-bit units,
    // so the 2^-23 is pre-folded (exact compile-time fold; identical value to
    // the production init's fold of the same two factors).
    constexpr float LOG2_EXPBIT = 0.693147182f * 1.19209290e-7f;
    // bf16-path minimax coefficients for the nonlinear correction of log1p(m)
    // on [-0.25, 0.5) (identical constants to production: they ARE the golden
    // polynomial).
    constexpr float C2 = -0x1p-2f;
    constexpr float C1 = 0x1.744p-2f;
    constexpr float C0 = -0x1.008p-1f;

    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat a = sfpi::dst_reg[0];
        const sfpi::vFloat u = a + 1.0f;
        // 1 + a < 0 lanes keep the default NaN.
        sfpi::vFloat r = std::numeric_limits<float>::quiet_NaN();

        v_if (u >= 0.0f)
        {
            // e = k << 23 for the unique k with 2^-k * u in [0.75, 1.5):
            // subtract the encoding of 0.75, then zero the mantissa field.
            const sfpi::vFloat three_quarters = 0.75f;
            sfpi::vInt e                      = sfpi::as<sfpi::vInt>(u) - sfpi::as<sfpi::vInt>(three_quarters);
            e                                 = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));

            // 2^-k * a by the same encoding arithmetic; t = 2^-k - 1 exactly,
            // through a 4.0 anchor (4 * 2^-k stays normal over the whole
            // reduced-exponent range, and both the 0.25 scale and the
            // subtraction of 1 are exact on these operands).
            const sfpi::vFloat four   = 4.0f;
            const sfpi::vFloat scaled = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
            const sfpi::vFloat s      = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(four) - e);
            const sfpi::vFloat m      = scaled + (0.25f * s - 1.0f); // 2^-k * (1 + a) - 1

            // k*ln(2): |e| = |k| << 23 < 2^31 converts exactly; restore sign.
            sfpi::vFloat e_float = sfpi::convert<sfpi::vFloat>(sfpi::abs(e), sfpi::RoundMode::Nearest);
            e_float              = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));

            r = C2 * m + C1;
            r = r * m + C0;
            r = r * (m * m) + m;
            r = e_float * LOG2_EXPBIT + r;

            // u >= 0 here, so encoding order finds exactly +inf and NaN
            // inputs: both pass through (log1p(inf) = inf, log1p(nan) = nan).
            const sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
            v_if (sfpi::as<sfpi::vInt>(u) >= sfpi::as<sfpi::vInt>(infinity))
            {
                r = u;
            }
            v_endif;
        }
        v_endif;

        // bf16 destination: round to nearest-even before the store truncates.
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

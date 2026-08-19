// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// expm1 — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// depends on fresh_round_nearest, which stays in
// fresh_cpp_operations.h (shared with the legacy remainder-family bodies).
#include <cstdint>

namespace ckernel::sfpu
{

// Expm1, bf16 contract (production: metal _sfpu_expm1_ non-fp32 branch —
// Juffa reduction with log2e / -ln2 / c1 parked in vConstFloatPrgm0/1/2, two
// raw __builtin_rvtt_sfpmad pins, and hand-interleaved Horner).  Identical
// arithmetic, plain statement: i = rint(a/ln2), f = a - i*ln2, quartic
// expm1(f), half-scaled 2^i reconstruction, saturation via the SMag8 clamp.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_expm1_fresh_cpp()
{
    constexpr float LOG2E   = 1.442695f;
    constexpr float NEG_LN2 = -0.6931471805599453f;
    constexpr float C3      = 8.361816406e-03f;
    constexpr float C2      = 4.177856445e-02f;
    constexpr float C1      = 1.666259766e-01f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vInt i;
        const sfpi::vFloat j = fresh_round_nearest(a * LOG2E, i);
        const sfpi::vFloat f = j * NEG_LN2 + a;

        sfpi::vFloat r       = C3;
        r                    = r * f + C2;
        r                    = r * f + C1;
        r                    = r * f + 0.5f;
        const sfpi::vFloat s = f * f;
        r                    = r * s + f;

        // For j == 0, r already is expm1(a); the half-scaled reconstruction
        // would flush tiny normal results through a subnormal.
        v_if (j != 0.0f)
        {
            const sfpi::vFloat w     = 0.5f;
            const sfpi::vFloat scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w)); // 0.5 * 2^i
            const sfpi::vFloat bias  = scale - w;
            const sfpi::vFloat jm2   = j + -2.0f;
            r                        = scale * r + bias;

            // Saturation: |i - 2| >= 127 covers a*log2(e) <= -125 / >= 129.
            const sfpi::vInt tail = sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(sfpi::abs(jm2), sfpi::RoundMode::Nearest));
            v_if (tail >= 127)
            {
                // +inf on the positive side; NaN propagates through the multiply.
                r = jm2 * std::numeric_limits<float>::infinity();
                v_if (jm2 < 0.0f)
                {
                    r = -0.5f;
                }
                v_endif;
            }
            v_endif;
            r = r * 2.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

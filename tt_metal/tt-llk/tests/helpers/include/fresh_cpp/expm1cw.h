// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// expm1cw — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// depends on fresh_round_nearest, which stays in
// fresh_cpp_operations.h (shared with the legacy remainder-family bodies).
#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

// Component-wise expm1 (production: tt-llk expm1_cw_clamped — Cody-Waite with
// the raw 0x4B400000 rounding-bias constant and the fused 0x4B3FFF81 ISUB;
// looped by a test adapter).  Same reduction, polynomials, and clamp; the
// round-nearest and the 2^k reconstruction stated typed.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_expm1_cw_fresh_cpp()
{
    constexpr float INV_LN2    = 1.4426950408889634f;
    constexpr float LN2_HI_NEG = -0.6931152343750000f;
    constexpr float LN2_LO_NEG = -3.19461832987e-05f;
    // Largest x whose expm1 is a finite fp32 (exp(88.7229) = FLT_MAX); above it
    // the value is +inf, stated rather than computed (the reduction has no upper
    // bound and setexp's 8-bit field wraps instead of saturating).
    constexpr float EXPM1_MAX  = 88.5f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x_raw = sfpi::dst_reg[0];
        sfpi::vFloat x           = sfpi::max(x_raw, -87.0f);
        x                        = sfpi::min(x, EXPM1_MAX);

        sfpi::vInt k_int;
        // Cap k at 127: round-nearest gives k = 128 on (127.5*ln2, 88.5], where 2^k
        // is not a finite fp32 (inf - inf = NaN at 88.5). There r reaches 0.4703,
        // past the fit range; the fits' error is 2.3e-7 / 4.1e-6 relative, under
        // half a bf16 ulp. One min, no branch; k <= 127 lanes are untouched.
        const sfpi::vFloat k = fresh_round_nearest(sfpi::min(x * INV_LN2, 127.0f), k_int);
        sfpi::vFloat r       = k * LN2_HI_NEG + x;
        r                    = r + k * LN2_LO_NEG;

        // expm1(r) = r * h(r) (production Sollya fits per format arm).
#ifdef INP_FLOAT32
        sfpi::vFloat h = 1.3948583510e-03f;
        h              = h * r + 8.3691505715e-03f;
        h              = h * r + 4.1666239500e-02f;
        h              = h * r + 1.6666504741e-01f;
        h              = h * r + 5.0000000000e-01f;
        h              = h * r + 1.0f;
#else
        sfpi::vFloat h = 8.3751315251e-03f;
        h              = h * r + 4.1875664145e-02f;
        h              = h * r + 1.6666433215e-01f;
        h              = h * r + 4.9999371171e-01f;
        h              = h * r + 1.0f;
#endif
        h = r * h;

        const sfpi::vFloat two_k = sfpi::setexp(sfpi::vFloat(1.0f), k_int + 127);
        sfpi::vFloat result      = (two_k - 1.0f) + two_k * h;
        v_if (x_raw > EXPM1_MAX)
        {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
        // NaN in, NaN out (max/min above send a NaN to -87 or 88.5). Clear the sign
        // with an integer AND -- SFPABS leaves a NaN's sign alone -- and one integer
        // compare: |bits| > 0x7F800000 holds for exactly the NaN patterns.
        v_if ((sfpi::as<sfpi::vInt>(x_raw) & 0x7FFFFFFF) > 0x7F800000)
        {
            result = x_raw;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

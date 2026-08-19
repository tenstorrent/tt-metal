// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// expm1cw — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// depends on fresh_round_nearest, which stays in
// fresh_cpp_operations.h (shared with the legacy remainder-family bodies).
#include <cstdint>

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
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x              = sfpi::max(x, -87.0f);

        sfpi::vInt k_int;
        const sfpi::vFloat k = fresh_round_nearest(x * INV_LN2, k_int);
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
        sfpi::dst_reg[0]         = (two_k - 1.0f) + two_k * h;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

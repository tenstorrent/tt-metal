// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the xielu op (storm contract:
// fresh_cpp/README.md).  Migrated verbatim from fresh_cpp_operations.h
// (Lane BR batch 2); byte-stable algorithm, only the file moved.
//
// NATURAL-FORM STATUS (storm S5, pin 12): the natural typed statement —
// alpha_p/alpha_n held as loop-invariant vFloat locals — now COMPILES at
// the ON flag set (const-remat landed; the pin-10 RA ICE is gone), but the
// sweep's OFF leg (-mno-tt-tensix-optimize-const-remat/-const-residency)
// refuses it with the clean "lreg-pressure-exceeded" diagnostic that names
// those flags as the fix.  A full2x2 row needs both legs to compile, so the
// per-use host-float materialization below stays canonical until the passes
// are default-ON in the shipped compiler (owner order 2026-08-18); then the
// natural form becomes the semantic statement.

#include <cstdint>

// Shared helpers (fresh_round_nearest) still live in the legacy header
// pending full migration (fresh_cpp/README.md legacy note).
#include "fresh_cpp_operations.h"

namespace ckernel::sfpu
{

// xIELU (production: metal calculate_xielu — eps and expm1(eps) hidden in
// vConstFloatPrgm1/2 and read back inside v_elseif conditions, plus a
// private Cody-Waite negative-exp with 1/ln2 in vConstFloatPrgm0).  Same
// four-region contract and constants (all the golden math), stated with
// plain locals; alpha_p/alpha_n arrive exactly as the production dispatch
// sends them, beta = 0.5 fixed.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_xielu_fresh_cpp(const std::uint32_t param0, const std::uint32_t param1)
{
    constexpr float EPS        = -1e-6f;
    constexpr float EXPM1_EPS  = -0.0000009999995427f;
    constexpr float ONE_LN2    = 1.4426950408889634f;
    constexpr float LN2_HI_NEG = -0.6931152343750000f;
    constexpr float LN2_LO_NEG = -3.19461832987e-05f;
    // Sollya expm1 tail (production constants), and the Taylor exp septic.
    constexpr float T2 = 0.500000059604644775390625f;
    constexpr float T3 = 0.16666667163372039794921875f;
    constexpr float T4 = 4.16650883853435516357421875e-2f;
    constexpr float T5 = 8.333188481628894805908203125e-3f;
    constexpr float T6 = 1.400390756316483020782470703125e-3f;
    constexpr float T7 = 1.99588379473425447940826416015625e-4f;

    // Plain host floats: materialized per use, not held in vector registers
    // across the loop (see the NATURAL-FORM STATUS note above — the vector
    // loop-held form refuses at the OFF flag set).
    const float alpha_p = Converter::as_float(param0);
    const float alpha_n = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x          = sfpi::dst_reg[0];
        const sfpi::vFloat beta_mul_x = 0.5f * x;
        // Positive arm as the all-lane default; the negative regions overwrite.
        sfpi::vFloat result = (alpha_p * x) * x + beta_mul_x;
        v_if (x <= 0.0f && x >= EPS)
        {
            result = alpha_n * (EXPM1_EPS - x) + beta_mul_x;
        }
        v_elseif (x<0.0f && x> - 0.5f)
        {
            // expm1(x) - x = x^2 * (T2 + T3 x + ... + T7 x^5) (cancellation-free).
            sfpi::vFloat p = T7;
            p              = p * x + T6;
            p              = p * x + T5;
            p              = p * x + T4;
            p              = p * x + T3;
            p              = p * x + T2;
            result         = alpha_n * (x * x * p) + beta_mul_x;
        }
        v_elseif (x <= -0.5f)
        {
            // exp(x) - 1 - x for large negative x, exp by Cody-Waite reduction
            // and the Taylor septic (production constants).
            sfpi::vFloat z = x * ONE_LN2;
            z              = sfpi::max(z, -126.5f);
            sfpi::vInt k_int;
            const sfpi::vFloat k = fresh_round_nearest(z, k_int);
            const sfpi::vFloat r = k * LN2_LO_NEG + (k * LN2_HI_NEG + x);

            sfpi::vFloat p = 1.0f / 5040.0f;
            p              = p * r + 1.0f / 720.0f;
            p              = p * r + 1.0f / 120.0f;
            p              = p * r + 1.0f / 24.0f;
            p              = p * r + 1.0f / 6.0f;
            p              = p * r + 0.5f;
            p              = p * r + 1.0f;
            p              = p * r + 1.0f;

            const sfpi::vFloat expx = sfpi::setexp(p, sfpi::exexp(p, sfpi::ExponentMode::Biased) + k_int);
            result                  = alpha_n * (expx - 1.0f - x) + beta_mul_x;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

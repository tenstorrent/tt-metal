// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: softplus (llk_lib/vec/common.h has the shape of a function file).
// log(1 + exp(beta x)) / beta, and x above the threshold.
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h.
// Fp32 (a 32-bit Dest) picks the residual polynomial: degree 8 with an exp tail there, degree 6 for bf16.

#pragma once

#include <cstdint>
#include <type_traits>

#include "llk_lib/vec/common.h"
#include "llk_lib/vec/exp.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

// ======================================================================
// Softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// FP32: degree-8 polynomial for f(a) on [0, 5] + inline exp + 3-term Taylor tail
// BF16: degree-6 polynomial (bf16-accurate, <0.28 ULP) + tail clamped to 0
//       (residual < exp(-5) = 0.0067 for a > 5, below bf16 rounding vs the t>0 term,
//        so the expensive exp tail is unnecessary at bf16 precision)
// ======================================================================

constexpr float vec_softplus_poly_boundary = 5.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float vec_softplus_poly_c0 = 6.9310557842e-01f;
constexpr float vec_softplus_poly_c1 = -4.9926245213e-01f;
constexpr float vec_softplus_poly_c2 = 1.2186349183e-01f;
constexpr float vec_softplus_poly_c3 = 5.6753782555e-03f;
constexpr float vec_softplus_poly_c4 = -1.0528374463e-02f;
constexpr float vec_softplus_poly_c5 = 2.7290175203e-03f;
constexpr float vec_softplus_poly_c6 = -3.4358495031e-04f;
constexpr float vec_softplus_poly_c7 = 2.1285692128e-05f;
constexpr float vec_softplus_poly_c8 = -4.8245715334e-07f;

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 6
// (ULP-weighted minimax fit; max error < 0.28 bf16 ULP over the domain)
constexpr float vec_softplus_bf16_poly_c0 = 6.9423984729e-01f;
constexpr float vec_softplus_bf16_poly_c1 = -5.0932420424e-01f;
constexpr float vec_softplus_bf16_poly_c2 = 1.4279095486e-01f;
constexpr float vec_softplus_bf16_poly_c3 = -1.3000584069e-02f;
constexpr float vec_softplus_bf16_poly_c4 = -1.8627923291e-03f;
constexpr float vec_softplus_bf16_poly_c5 = 5.0152968088e-04f;
constexpr float vec_softplus_bf16_poly_c6 = -3.1273466851e-05f;

// ======================================================================
// Lightweight inline exp(x) for negative x (tail region).
// Adapted from x_times_exp_negative_tail of 1.0's gelu (ckernel_sfpu_gelu.h).
// Uses Cody-Waite range reduction + Taylor polynomial.
// BF16: degree 5 (~15 ops), FP32: degree 7 (~19 ops).
// ======================================================================
template <bool Fp32>
sfpi_inline sfpi::vFloat vec_softplus_exp_negative(sfpi::vFloat x)
{
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI  = -0.6931152343750000f;
    constexpr float LN2_LO  = -3.19461832987e-05f;

    // Range reduction: x = k*ln(2) + r
    sfpi::vFloat z = x * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = vec_round_to_nearest_int(z, k_int);

    // Cody-Waite: r = x - k*ln(2) in extended precision
    sfpi::vFloat r = k * LN2_HI + x;
    r              = k * LN2_LO + r;

    // exp(r) via Taylor polynomial, |r| < 0.5
    sfpi::vFloat poly;
    if constexpr (Fp32)
    {
        // FP32: degree 7 for < 1 ULP
        poly = vec_polynomial::eval(r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);
    }
    else
    {
        // BF16: degree 5 sufficient
        poly = vec_polynomial::eval(r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f);
    }

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp   = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if (new_exp > 0)
    {
        result = sfpi::setexp(poly, new_exp);
    }
    v_endif;

    return result;
}

// The three lowest residual-polynomial coefficients of Fp32's fit, which vec_softplus_init parks in
// vConstFloatPrgm0/1/2 for vec_softplus_iterations (this kernel runs no reciprocal, so Prgm0 is free).
template <bool Fp32>
constexpr float vec_softplus_prgm_c0 = Fp32 ? vec_softplus_poly_c0 : vec_softplus_bf16_poly_c0;
template <bool Fp32>
constexpr float vec_softplus_prgm_c1 = Fp32 ? vec_softplus_poly_c1 : vec_softplus_bf16_poly_c1;
template <bool Fp32>
constexpr float vec_softplus_prgm_c2 = Fp32 ? vec_softplus_poly_c2 : vec_softplus_bf16_poly_c2;

/// softplus's part of vec_configure: the residual polynomial's three lowest coefficients. Fast is not read.
template <bool Fast, bool Fp32>
inline void vec_softplus_init()
{
    sfpi::vConstFloatPrgm0 = vec_softplus_prgm_c0<Fp32>;
    sfpi::vConstFloatPrgm1 = vec_softplus_prgm_c1<Fp32>;
    sfpi::vConstFloatPrgm2 = vec_softplus_prgm_c2<Fp32>;
}

// One row. The scalars and the polynomial's three lowest coefficients are supplied by the caller: for
// vec_softplus_iterations, beta and threshold (and beta_reciprocal unless Fp32, see there) are sfpi::vFloats
// built once before its loop (a runtime float used per row otherwise costs RISC-built SFPLOADIs on every row)
// and the coefficients are vConstFloatPrgm0/1/2 from vec_softplus_init. Identical arithmetic either way (hoisted
// constants: llk_lib/vec/common.h).
template <bool Fast, bool Fp32, typename B, typename BR, typename T, typename C0, typename C1, typename C2>
inline void vec_softplus_row(B beta, BR beta_reciprocal, T threshold, C0 c0, C1 c1, C2 c2)
{
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t   = beta * val;

    v_if (t <= threshold)
    {
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

        sfpi::vFloat residual;
        if constexpr (Fp32)
        {
            // FP32: f(a) via degree-8 Horner on [0, 5]
            residual = vec_polynomial::eval(
                a,
                c0,
                c1,
                c2,
                vec_softplus_poly_c3,
                vec_softplus_poly_c4,
                vec_softplus_poly_c5,
                vec_softplus_poly_c6,
                vec_softplus_poly_c7,
                vec_softplus_poly_c8);

            // Tail: f(a) ≈ exp(-a) for a > 5, via inline Cody-Waite exp +
            // 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
            sfpi::vFloat neg_a = sfpi::setsgn(a, 1);
            v_if (a > vec_softplus_poly_boundary)
            {
                sfpi::vFloat e = vec_softplus_exp_negative<Fp32>(neg_a);
                residual       = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;
        }
        else
        {
            // BF16: f(a) via degree-6 Horner on [0, 5]
            residual =
                vec_polynomial::eval(a, c0, c1, c2, vec_softplus_bf16_poly_c3, vec_softplus_bf16_poly_c4, vec_softplus_bf16_poly_c5, vec_softplus_bf16_poly_c6);

            // Tail: the degree-6 poly diverges past its [0, 5] fit domain, while the true
            // residual < exp(-5) = 0.0067 there. Clamping to 0 keeps softplus(t>0) = t within
            // bf16 rounding and avoids the ~8-op exp tail on every element.
            v_if (a > vec_softplus_poly_boundary)
            {
                residual = 0.0f;
            }
            v_endif;
        }

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        t               = sfpi::max(t, 0.0f);
        sfpi::vFloat sp = t + residual;

        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        sfpi::vFloat result = beta_reciprocal * sp;
        if constexpr (!Fp32)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

/// softplus on Iterations iterations; beta, 1 / beta and the threshold are FP32 bit patterns. Fast is not read:
/// softplus has one form.
template <bool Fast, bool Fp32, int Iterations = 8>
inline void vec_softplus_iterations(std::uint32_t beta_bits, std::uint32_t beta_reciprocal_bits, std::uint32_t threshold_bits)
{
    // The runtime scalars, converted to vectors once and kept in LRegs (per row they would each be
    // RISC-built SFPLOADIs); the polynomial's lowest coefficients come from Prgm0/1/2 (vec_softplus_init).
    // The Fp32 body (degree-8 poly plus the exp tail) leaves room for only two of them: a third loop-carried LReg
    // fails to compile ("too few lregs"), so beta_reciprocal stays a float there.
    using BetaReciprocal           = std::conditional_t<Fp32, float, sfpi::vFloat>;
    sfpi::vFloat beta              = vec_float_of(beta_bits);
    BetaReciprocal beta_reciprocal = vec_float_of(beta_reciprocal_bits);
    sfpi::vFloat threshold         = vec_float_of(threshold_bits);
    for (int d = 0; d < Iterations; d++)
    {
        vec_softplus_row<Fast, Fp32>(beta, beta_reciprocal, threshold, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2);
        sfpi::dst_reg++;
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

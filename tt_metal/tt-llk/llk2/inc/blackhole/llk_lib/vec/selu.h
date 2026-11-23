// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: selu (llk_lib/vec/common.h has the shape of a function file).
// scale * (x > 0 ? x : alpha * (exp(x) - 1)), the expm1 by Cody-Waite range reduction.
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h and ckernel_sfpu_expm1_cw.h.
// Fp32 (a 32-bit Dest) picks the expm1 polynomial: degree 5 there, degree 4 for bf16.

#pragma once

#include <cstdint>

#include "llk_lib/vec/common.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

// ======================================================================
// Shared helper: exp(x) - 1 via Cody-Waite range reduction + factored
// expm1 polynomial. In LLK 2.0 selu uses it.
//
// Algorithm: x = k*ln(2) + r, |r| <= ln(2)/2
//   expm1(r) = r * h(r), h(r) = minimax polynomial on [-ln2/2, ln2/2]
//   exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
//
// BF16 h degree 4: max abs error = 1.60e-7 (Sollya remez)
// FP32 h degree 5: max abs error = 8.67e-9 (Sollya remez)
// ======================================================================

constexpr float vec_cw_inv_ln2    = 1.4426950408889634f;
constexpr float vec_cw_neg_ln2_hi = -0.6931152343750000f;
constexpr float vec_cw_neg_ln2_lo = -3.19461832987e-05f;

// The two highest-order Horner coefficients of h(r), exported so a row-loop caller can hold them in LRegs
// (see the hoisted-constant overload below). Which polynomial they belong to follows Fp32, as the evaluation
// itself does.
template <bool Fp32>
constexpr float vec_cw_expm1_h_top1 = Fp32 ? 8.3691505715e-03f : 4.1875664145e-02f;
template <bool Fp32>
constexpr float vec_cw_expm1_h_top0 = Fp32 ? 1.3948583510e-03f : 8.3751315251e-03f;

// vec_expm1_clamped with its five loop-invariant fp32 constants supplied by the caller. Each may be a float
// (materialised by SFPLOADI where used, exactly as the literal is in vec_expm1_clamped(x)) or an
// sfpi::vFloat / vConstFloatPrgmN the caller holds across its row loop (hoisted constants: llk_lib/vec/common.h).
// Identical arithmetic either way. The constants are template-typed rather than sfpi::vFloat because a
// vFloat parameter is built before the call and stays live through the body, which changes the
// non-hoisting callers' code.
template <bool Fp32, typename K, typename HI, typename LO, typename T1, typename T0>
sfpi_inline sfpi::vFloat vec_expm1_clamped(sfpi::vFloat x, K inv_ln2, HI neg_ln2_hi, LO neg_ln2_lo, T1 h_top1, T0 h_top0)
{
    // Clamp to prevent exponent underflow (k < -127 wraps setexp)
    x = sfpi::max(x, -87.0f);

    // Cody-Waite range reduction: x = k*ln(2) + r
    const sfpi::vFloat c231 = vec_float_of(0x4B400000U);
    sfpi::vFloat tmp        = x * inv_ln2 + c231;
    sfpi::vFloat k_f        = tmp - c231;
    sfpi::vFloat r          = k_f * neg_ln2_hi + x;
    r                       = r + k_f * neg_ln2_lo;

    // expm1(r) = r * h(r), Horner evaluation of h
    sfpi::vFloat h;
    if constexpr (Fp32)
    {
        h = vec_polynomial::eval(r, 1.0f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, h_top1, h_top0);
    }
    else
    {
        h = vec_polynomial::eval(r, 1.0f, 4.9999371171e-01f, 1.6666433215e-01f, h_top1, h_top0);
    }
    h = r * h;

    // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
    // 0x4B3FFF81 = 0x4B400000 - 127: fuses k_int ISUB + bias IADD into a single ISUB
    constexpr int kC231Bias = 0x4B3FFF81;
    sfpi::vFloat two_k      = sfpi::setexp(1.0f, sfpi::as<sfpi::vInt>(tmp) - kC231Bias);
    return (two_k - 1.0f) + two_k * h;
}

template <bool Fp32>
sfpi_inline sfpi::vFloat vec_expm1_clamped(sfpi::vFloat x)
{
    return vec_expm1_clamped<Fp32>(x, vec_cw_inv_ln2, vec_cw_neg_ln2_hi, vec_cw_neg_ln2_lo, vec_cw_expm1_h_top1<Fp32>, vec_cw_expm1_h_top0<Fp32>);
}

/**
 * @brief Programs the three Cody-Waite constants into vConstFloatPrgm0/1/2 for vec_expm1_clamped_prgm.
 *
 * Only for kernels that run no reciprocal: Prgm0 is vec_recip_newton_init's 2.0f wherever one does.
 *
 * @note Call this in the kernel's init; @ref vec_expm1_clamped_prgm reads vConstFloatPrgm0-2 on every row.
 */
inline void vec_expm1_init()
{
    sfpi::vConstFloatPrgm0 = vec_cw_inv_ln2;
    sfpi::vConstFloatPrgm1 = vec_cw_neg_ln2_hi;
    sfpi::vConstFloatPrgm2 = vec_cw_neg_ln2_lo;
}

/**
 * @brief Row-loop form of vec_expm1_clamped: the Cody-Waite constants come from vConstFloatPrgm0/1/2, the two
 *        top Horner coefficients from the caller.
 *
 * h_top1 / h_top0 are an sfpi::vFloat built once before the caller's loop, or a float literal where its LRegs
 * are all spoken for (sfpi cannot spill: one hoisted constant too many is a compile error, "too few lregs",
 * never a slowdown).
 *
 * @param x: Input value.
 * @param h_top1: vec_cw_expm1_h_top1, bound by the caller outside its row loop.
 * @param h_top0: vec_cw_expm1_h_top0, bound by the caller outside its row loop.
 * @note Call @ref vec_expm1_init before this function; it reads vConstFloatPrgm0-2.
 */
template <bool Fp32, typename T1, typename T0>
sfpi_inline sfpi::vFloat vec_expm1_clamped_prgm(sfpi::vFloat x, T1 h_top1, T0 h_top0)
{
    return vec_expm1_clamped<Fp32>(x, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, h_top1, h_top0);
}

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

// Programs the Cody-Waite constants of vec_expm1_clamped into vConstFloatPrgm0/1/2, which vec_selu_iterations
// reads on every row. This kernel runs no reciprocal, so Prgm0 is not vec_recip_newton_init's 2.0f here.
template <bool Fast, bool Fp32>
inline void vec_selu_init()
{
    vec_expm1_init();
}

/// selu on Iterations iterations; scale and alpha are FP32 bit patterns. Fast is not read: selu has one form (the
/// Resolver refuses its fast form, resolver/vec.h).
template <bool Fast, bool Fp32, int Iterations = 8>
inline void vec_selu_iterations(std::uint32_t scale_bits, std::uint32_t alpha_bits)
{
    const sfpi::vFloat scale_val   = vec_float_of(scale_bits);
    const sfpi::vFloat scale_alpha = vec_float_of(scale_bits) * vec_float_of(alpha_bits);
    // expm1's top Horner coefficient, loaded once and held in an LReg for the whole loop (the three
    // Cody-Waite constants come from Prgm0/1/2, see the init above; hoisted constants: llk_lib/vec/common.h).
    // With scale_val live as well there is no LReg for the second one, so it stays a per-row literal.
    sfpi::vFloat h_top1    = vec_cw_expm1_h_top1<Fp32>;
    constexpr float h_top0 = vec_cw_expm1_h_top0<Fp32>;
// unroll 2: with vec_expm1_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat x      = sfpi::dst_reg[0];
        sfpi::vFloat result = scale_alpha * vec_expm1_clamped_prgm<Fp32>(x, h_top1, h_top0);

        v_if (x >= 0.0f)
        {
            result = scale_val * x;
        }
        v_endif;

        if constexpr (!Fp32)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: sigmoid (llk_lib/vec/common.h has the shape of a function file).
// 1 / (1 + exp(-x)): the exp and a Newton reciprocal, or a 3-segment lookup (fast).
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h and ckernel_sfpu_sigmoid_appx.h.

#pragma once

#include "llk_lib/vec/common.h"
#include "llk_lib/vec/exp.h"
#include "llk_lib/vec/recip.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

template <int Iterations = 8>
inline void vec_sigmoid_lut_iterations()
{
    sfpi::vLut8si si0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut8si si1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut8si si2 = sfpi::l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = sfpi::lut(val, si0, si1, si2) + 0.5f;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = si0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = si1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = si2;
}

inline void vec_sigmoid_lut_init()
{
    // Load the 3 fp16b LUT coefficients into LReg0-2
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 0.5f);
}

// sigmoid(x) = 1 / (1 + exp(-x)) with the exp's constants supplied by the caller. Each is a float (materialised
// by SFPLOADI where used, exactly as a literal is), an sfpi::vFloat built once before a row loop, or a
// vConstFloatPrgmN: one_ln2 is read by both exp arms, (c0, c1, c2) by the bf16 exp_21f arm and
// (neg_ln2_hi, p0, p1) by the fp32 Juffa arm (hoisted constants: llk_lib/vec/common.h).
// vec_sigmoid_value below is a thin wrapper over this body.
template <bool Fp32, typename K, typename C0, typename C1, typename C2, typename H, typename P0, typename P1>
sfpi_inline sfpi::vFloat vec_sigmoid_core(sfpi::vFloat x, K one_ln2, C0 c0, C1 c1, C2 c2, H neg_ln2_hi, P0 p0, P1 p1)
{
    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (Fp32)
    {
        exp_neg_x = vec_exp_fp32<false /*unsafe*/>(-x, one_ln2, neg_ln2_hi, p0, p1);
    }
    else
    {
        exp_neg_x = vec_exp_21f<true /*Fp32*/>(-x, one_ln2, c0, c1, c2);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (Fp32)
    {
        result = vec_recip_newton<2 /*max_iter*/>(denominator);
    }
    else
    {
        result = vec_recip_newton<1 /*max_iter*/>(denominator);
    }

    return result;
}

// Row-loop form: 1/ln2 from vConstFloatPrgm1 (programmed by vec_sigmoid_init<false>, which the caller's init
// must have run), the other exp constants from the caller (see vec_sigmoid_core).
template <bool Fp32, typename C0, typename C1, typename C2, typename H, typename P0, typename P1>
sfpi_inline sfpi::vFloat vec_sigmoid_value(sfpi::vFloat x, C0 c0, C1 c1, C2 c2, H neg_ln2_hi, P0 p0, P1 p1)
{
    return vec_sigmoid_core<Fp32>(x, sfpi::vConstFloatPrgm1, c0, c1, c2, neg_ln2_hi, p0, p1);
}

/// 1 / (1 + exp(-x)) on Iterations iterations: the exp and a Newton reciprocal, or (Fast) the 3-segment lookup.
template <bool Fast, bool Fp32, int Iterations = 8>
inline void vec_sigmoid_iterations()
{
    if constexpr (!Fast)
    {
        // The exp's constants, loaded once and kept in LRegs for the whole loop; only the arm's own set is
        // hoisted (the other stays a float). Nothing but the row is live, so every constant fits.
        vec_hoisted_if<!Fp32> c0 = vec_exp_21f_c0, c1 = vec_exp_21f_c1, c2 = vec_exp_21f_c2;
        vec_hoisted_if<Fp32> neg_ln2_hi = vec_exp_fp32_neg_ln2_hi, p0 = vec_exp_fp32_p0, p1 = vec_exp_fp32_p1;
#pragma GCC unroll 8
        for (int d = 0; d < Iterations; d++)
        {
            sfpi::vFloat val    = sfpi::dst_reg[0];
            sfpi::vFloat result = vec_sigmoid_value<Fp32>(val, c0, c1, c2, neg_ln2_hi, p0, p1);
            if constexpr (!Fp32)
            {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
    else
    {
        vec_sigmoid_lut_iterations<Iterations>();
    }
}

// What the exact form's init programs for vec_sigmoid_iterations, and for vec_silu_iterations (vec_silu_init
// forwards here):
//   Prgm0 = 2.0f    (vec_recip_newton_init; the Newton step in vec_recip_newton)
//   Prgm1 = 1/ln2   (both exps read it; 0x3FB8AA3B)
// Prgm2 is not used (see vec_exp_one_ln2_init). The fast form programs none of these: its lookup table is in
// L0..L2. vec_configure runs the init directly in front of the iterations of every call, so nothing else writes
// these registers in between.
template <bool Fast, bool Fp32>
inline void vec_sigmoid_init()
{
    if constexpr (!Fast)
    {
        vec_recip_newton_init<false>();
        vec_exp_one_ln2_init();
    }
    else
    {
        vec_sigmoid_lut_init();
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

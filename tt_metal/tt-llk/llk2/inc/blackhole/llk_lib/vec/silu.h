// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: silu (llk_lib/vec/common.h has the shape of a function file).
// x * sigmoid(x). Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h.

#pragma once

#include "llk_lib/vec/common.h"
#include "llk_lib/vec/exp.h"
#include "llk_lib/vec/sigmoid.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

/// x * sigmoid(x) on Iterations iterations, the sigmoid in its exact form. Fast is not read: silu has one form.
template <bool Fast, bool Fp32, int Iterations>
inline void vec_silu_iterations()
{
    // The exp's constants, loaded once and kept in LRegs for the whole loop (see vec_sigmoid_value); 1/ln2 is
    // Prgm1 from vec_silu_init. x stays live across the sigmoid, so the fp32 arm has room for two of its
    // three: p1 stays a per-row literal.
    vec_hoisted_if<!Fp32> c0 = vec_exp_21f_c0, c1 = vec_exp_21f_c1, c2 = vec_exp_21f_c2;
    vec_hoisted_if<Fp32> neg_ln2_hi = vec_exp_fp32_neg_ln2_hi, p0 = vec_exp_fp32_p0;
    constexpr float p1 = vec_exp_fp32_p1;
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * vec_sigmoid_value<Fp32>(x, c0, c1, c2, neg_ln2_hi, p0, p1);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!Fp32)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

/// silu's part of vec_configure: the exact sigmoid's constants. Fast and Fp32 are not read.
template <bool Fast, bool Fp32>
inline void vec_silu_init()
{
    // vec_silu_iterations uses the non-approx sigmoid path via vec_sigmoid_value, so we must use non-approx
    // vec_sigmoid_init: it programs Prgm0 = 2.0f for the reciprocal and Prgm1 = 1/ln2 for the exp.
    vec_sigmoid_init<false /*Fast*/, Fp32>();
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

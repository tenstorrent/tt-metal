// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: hardsigmoid (llk_lib/vec/common.h has the shape of a function file).
// min(max(x / 6 + 1 / 2, 0), 1). Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_activations.h.

#pragma once

#include "llk_lib/vec/common.h"
#include "llk_lib/vec/relu_max.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

/// min(max(v / 6 + 1 / 2, 0), 1) on one value; the slope and the offset are Prgm0 and Prgm1 (vec_hardsigmoid_init).
inline void vec_hardsigmoid_apply(sfpi::vFloat& v)
{
    sfpi::vFloat tmp = (v * sfpi::vConstFloatPrgm0) + sfpi::vConstFloatPrgm1;
    // clamp(tmp, 0, 1) as two SFPSWAP folds against the constant registers (threshold first,
    // then the relu clamp, so a NaN lands on 1.0): 7 instructions per row, 10 as two v_if
    // blocks. Kept as the shared relu_max body rather than sfpi::clamp, which folds in the
    // other order and would change what a NaN or a -0.0 lane returns.
    v = vec_relu_max_value(tmp, 1.0f);
}

/// hardsigmoid on Iterations iterations. One form: Fast and Fp32 are not read.
template <bool Fast, bool Fp32, int Iterations>
inline void vec_hardsigmoid_iterations()
{
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        vec_hardsigmoid_apply(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

/// hardsigmoid's part of vec_configure: the slope and the offset in Prgm0 and Prgm1. Fast and Fp32 are not read.
template <bool Fast, bool Fp32>
void vec_hardsigmoid_init()
{
    // For hardsigmoid slope is 1/6, FP32 IEEE 754 representation.
    sfpi::vConstFloatPrgm0 = 0.1666666716337204f;
    sfpi::vConstFloatPrgm1 = 0.5f;
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

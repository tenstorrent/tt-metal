// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: relu_max (llk_lib/vec/common.h has the shape of a function file).
// min(max(x, 0), threshold). Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h.

#pragma once

#include <cstdint>

#include "llk_lib/vec/common.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

// relu_max(x) = max(min(x, threshold), 0): two SFPSWAP min/max folds in place of two predicated
// blocks (SFPGT/SFPMOV/SFPENCC each), 9 -> 5 instructions per row when `threshold` is a literal or
// a constant register, 6 when it lives in a general LReg (SFPSWAP writes both operands, so the
// compiler copies the threshold first).
//
// The order is the one the predicated chain had -- threshold first, then the relu clamp -- and
// it is not interchangeable with min(max(x, 0), threshold): for threshold < 0 this form returns
// +0.0 on every lane (min(x, t) <= t < 0, then the clamp), the other would return the threshold.
// SFPSWAP orders FP32 by sign-magnitude, -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN, so
//   +NaN -> threshold (outranks every value, the min lands on the threshold)
//   -NaN -> +0.0      (ranks below every value, survives the min, the clamp takes it)
//   -0.0 -> +0.0      (ranks below +0.0)
// which is what the SFPGT/SFPSETCC chain returned as well; measured on Blackhole silicon, see 1.0's
// test_eltwise_unary_sfpu_relu_max_threshold.
sfpi_inline sfpi::vFloat vec_relu_max_value(sfpi::vFloat val, sfpi::vFloat threshold)
{
    return sfpi::max(sfpi::min(val, threshold), 0.0f);
}

// Float relu_max with a per-call scalar threshold. The threshold is parked in vConstFloatPrgm1
// (LREG13) for the duration of the call: SFPSWAP writes both of its operands, so a threshold held
// in a general LReg costs an SFPMOV copy per row, while a write to a constant register is dropped.
// One SFPCONFIG (+2 SFPLOADI) per call buys one instruction per row: 5 instead of 6, 9 before.
//
// Prgm1 is call-scoped here: vec_relu_max_init programs nothing, each call of the iterations (one per face)
// writes it again, and vec_configure runs every function's init in front of its own iterations, so nothing relies
// on Prgm1 surviving a relu_max call.
//
// The SFPI compiler compiles exactly this shape -- max(min(v, Prgm), literal) -- to SFPSWAP against
// LREG13 and LREG9; the bare min/max/clamp(v, vFloat(vConstFloatPrgmN)) forms hit an internal
// compiler error (lra-constraints; seen with sfpi 7.83.0), so do not "simplify" this into vec_relu_max_value.
template <int Iterations>
inline void vec_relu_max_float_iterations(const float threshold)
{
    sfpi::vConstFloatPrgm1 = threshold;
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::max(sfpi::min(v, sfpi::vFloat(sfpi::vConstFloatPrgm1)), 0.0f);
        sfpi::dst_reg++;
    }
}

/// min(max(x, 0), threshold) on Iterations iterations; the threshold is an FP32 bit pattern. One form: Fast and
/// Fp32 are not read.
template <bool Fast, bool Fp32, int Iterations>
inline void vec_relu_max_iterations(std::uint32_t threshold_bits)
{
    float f;
    f = vec_float_of(threshold_bits);
    vec_relu_max_float_iterations<Iterations>(f);
}

/// relu_max's part of vec_configure: nothing of its own. Fast and Fp32 are not read.
template <bool Fast, bool Fp32>
inline void vec_relu_max_init()
{
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

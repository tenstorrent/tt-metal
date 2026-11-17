// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Values match the Compute API's ADD_UNARY .. RSUB_UNARY and Blackhole's BINOP_MODE.
enum class BinopMode : int {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,  // the host passes 1 / scalar, so DIV multiplies
    Rsub = 4,
};

/**
 * @brief Elementwise float binop with a scalar on the Dest tile: y = binop(x, scalar), or scalar - x for Rsub.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP_MODE: the operation; Div expects the host to pass the reciprocal of the scalar
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 * @tparam is_fp32_dest_acc_en: with a 16-bit Dest, Rsub rounds its result to bf16 nearest-even, as on BH
 * @param param: fp32 scalar encoded as uint32
 */
template <bool APPROXIMATION_MODE, BinopMode BINOP_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_binop_with_scalar(std::uint32_t param) {
    const sfpi::vFloat parameter = __builtin_bit_cast(float, param);
    // SFPSTORE truncates fp32 to bf16. Rsub feeds logit(x) = log(x / (1 - x)), where that truncation error is
    // amplified by the log, so on a 16-bit Dest it rounds to nearest even the way torch computes bf16 rsub.
    constexpr bool round_to_bf16_rne = BINOP_MODE == BinopMode::Rsub && !is_fp32_dest_acc_en;
    // The RNE constants are bound once, not per row. Unassigned unless the rounding below uses them.
    sfpi::vUInt rne_bias;
    sfpi::vUInt rne_mask;
    if constexpr (round_to_bf16_rne) {
        rne_bias = kBf16RneBias;
        rne_mask = kBf16RneMask;
    }
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (BINOP_MODE == BinopMode::Add) {
            result = val + parameter;
        } else if constexpr (BINOP_MODE == BinopMode::Sub) {
            result = val - parameter;
        } else if constexpr (BINOP_MODE == BinopMode::Mul || BINOP_MODE == BinopMode::Div) {
            result = val * parameter;
        } else if constexpr (BINOP_MODE == BinopMode::Rsub) {
            result = parameter - val;
            if constexpr (round_to_bf16_rne) {
                result = float32_to_bf16_rne(result, rne_bias, rne_mask);
            }
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_mul(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, BinopMode::Mul, ITERATIONS, is_fp32_dest_acc_en>(param);
}

/**
 * @brief Int32 add with a scalar on the Dest tile, in two's complement: y = x + scalar.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 * @param scalar: int32 scalar encoded as uint32
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_add_int32(std::uint32_t scalar) {
    // The scalar is materialised into a vInt once; `a + s` lowers to a single SFPIADD per row.
    const sfpi::vInt s = static_cast<int>(scalar);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a + s;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Int32 subtract of a scalar on the Dest tile, in two's complement: y = x - scalar.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 * @param scalar: int32 scalar encoded as uint32
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_sub_int32(std::uint32_t scalar) {
    // Scalar hoisted once, as in calculate_add_int32; `a - s` lowers to a single SFPIADD per row.
    const sfpi::vInt s = static_cast<int>(scalar);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a - s;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

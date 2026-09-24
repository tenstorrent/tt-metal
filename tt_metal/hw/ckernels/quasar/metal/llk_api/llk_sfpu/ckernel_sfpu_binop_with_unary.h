// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

namespace ckernel::sfpu {

enum class BinopMode : int {
    Mul = 2,
};

template <bool APPROXIMATION_MODE, BinopMode BINOP_MODE, int ITERATIONS = SFPU_ITERATIONS>
void calculate_binop_with_scalar(std::uint32_t param) {
    static_assert(BINOP_MODE == BinopMode::Mul, "Quasar binop_with_scalar currently supports Mul (mode=2) only");
    const sfpi::vFloat parameter = __builtin_bit_cast(float, param);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (BINOP_MODE == BinopMode::Mul) {
            result = val * parameter;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
void calculate_mul(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, BinopMode::Mul, ITERATIONS>(param);
    return;
}

// Op class for an elementwise float binop with a scalar. Same name and leading template parameters as on
// Wormhole/Blackhole (BINOP_MODE: 0 add, 1 sub, 2 mul, 3 div, 4 rsub); Quasar supports mul (2) only.
template <
    bool APPROXIMATION_MODE,
    int BINOP_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool is_fp32_dest_acc_en = false,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct BinopWithScalar
    : SfpuUnaryOp<BinopWithScalar<APPROXIMATION_MODE, BINOP_MODE, ITERATIONS, is_fp32_dest_acc_en, SLOT>, SLOT> {
    static_assert(
        BINOP_MODE == static_cast<int>(BinopMode::Mul),
        "Quasar binop_with_scalar currently supports Mul (mode=2) only");
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t param) {
        calculate_binop_with_scalar<APPROXIMATION_MODE, static_cast<BinopMode>(BINOP_MODE), ITERATIONS>(param);
    }
};

}  // namespace ckernel::sfpu

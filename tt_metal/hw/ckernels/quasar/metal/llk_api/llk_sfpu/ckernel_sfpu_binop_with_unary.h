// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

enum class BinopMode : int {
    Mul = 2,
};

template <bool APPROXIMATION_MODE, BinopMode BINOP_MODE, int ITERATIONS = 8>
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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_mul(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, BinopMode::Mul, ITERATIONS>(param);
    return;
}

}  // namespace ckernel::sfpu

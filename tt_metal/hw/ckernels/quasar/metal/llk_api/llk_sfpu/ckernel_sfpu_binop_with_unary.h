// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_assert.h"
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

// Quasar has no UInt32 data format. The Compute API keeps one signature on every arch, so the kernel entry points
// of add_unary_tile_uint32 / sub_unary_tile_uint32 reject the call here.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_add_uint32([[maybe_unused]] std::uint32_t scalar) {
    LLK_ASSERT(false, "UInt32 unary add with scalar (add_unary_tile_uint32) is not supported on Quasar");
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sub_uint32([[maybe_unused]] std::uint32_t scalar) {
    LLK_ASSERT(false, "UInt32 unary sub with scalar (sub_unary_tile_uint32) is not supported on Quasar");
}

}  // namespace ckernel::sfpu

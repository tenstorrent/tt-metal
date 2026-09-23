// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

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

// ---------------------------------------------------------------------------------------------------
// BinopWithScalar<APPROX, BINOP_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>. Same interface as WH/BH.
//   BINOP_MODE is the compute API's integer mode (ADD_UNARY=0 .. RSUB_UNARY=4); Quasar's kernel currently
//   supports MUL_UNARY (== BinopMode::Mul) on float dest only.
//   calculate(dst_index, vector_mode, scalar) -> calculate_binop_with_scalar
//   init()                                    -> bare init (mode-independent)
// Backs mul_unary_tile, binop_with_scalar_tile_init and llk_math_eltwise_unary_sfpu_binop_with_scalar.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    int BINOP_MODE,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = SFPU_ITERATIONS>
struct BinopWithScalar : SfpuUnaryOp<
                             BinopWithScalar<APPROXIMATION_MODE, BINOP_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>,
                             DST_SYNC,
                             DST_ACCUM> {
    static_assert(FORMAT != DataFormat::Int32, "Quasar binop_with_scalar supports float dest only");

    static void kernel(std::uint32_t param) {
        calculate_binop_with_scalar<APPROXIMATION_MODE, static_cast<BinopMode>(BINOP_MODE), ITERATIONS>(param);
    }
};

}  // namespace ckernel::sfpu

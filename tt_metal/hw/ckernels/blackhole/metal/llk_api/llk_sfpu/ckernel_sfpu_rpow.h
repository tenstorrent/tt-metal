// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_binary_pow.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {
// ttnn.rpow(exponent, scalar_base) = pow(scalar_base, exponent)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_rpow(const uint32_t base_val) {
    sfpi::vFloat base_val_v = Converter::as_float(base_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _sfpu_binary_power_<is_fp32_dest_acc_en>(base_val_v, sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Rpow<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, base_val) -> calculate_rpow
//   init()                                      -> sfpu_binary_pow_init
// Backs rpow_tile / rpow_tile_init.
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Rpow : SfpuUnaryOp<Rpow<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t base_val) { calculate_rpow<APPROXIMATION_MODE, ITERATIONS, DST_ACCUM>(base_val); }

    static void init_kernel() { sfpu_binary_pow_init<APPROXIMATION_MODE>(); }
};
}  // namespace ckernel::sfpu

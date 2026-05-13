// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_binary_pow.h"
#include "sfpi.h"

namespace ckernel::sfpu {
// ttnn.rpow(exponent, scalar_base) = pow(scalar_base, exponent)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_rpow(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint32_t base_val) {
    sfpi::vFloat base_val_v = Converter::as_float(base_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[(dst_index_out - dst_index_in) * TILE_R_DIM] =
            _sfpu_binary_power_<is_fp32_dest_acc_en>(base_val_v, sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

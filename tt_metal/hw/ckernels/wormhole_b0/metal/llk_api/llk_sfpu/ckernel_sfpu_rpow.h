// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "ckernel_sfpu_binary.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en = false>
inline void calculate_rpow(const uint32_t base_val) {
    sfpi::vFloat base_val_v = Converter::as_float(base_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _sfpu_binary_power_<is_fp32_dest_acc_en>(base_val_v, sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

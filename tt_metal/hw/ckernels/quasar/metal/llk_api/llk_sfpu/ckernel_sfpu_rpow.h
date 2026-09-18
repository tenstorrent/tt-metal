// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_binary_pow.h"
#include "sfpi.h"
#include "llk_sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// ttnn.rpow(exponent, scalar_base) = pow(scalar_base, exponent)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_rpow(const uint32_t base_val) {
    const float base_val_f = Converter::as_float(base_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat result = _sfpu_binary_power_f32_magnitude_(base_val_f, sfpi::dst_reg[0]);
            // Dest has not been written: reload exponent instead of keeping it live through exp.
            result = _sfpu_binary_power_f32_finalize_(result, base_val_f, sfpi::dst_reg[0]);
            sfpi::dst_reg[0] = result;
        } else {
            sfpi::dst_reg[0] = _sfpu_binary_power_<false>(base_val_f, sfpi::dst_reg[0]);
        }
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_sfpu_binary_pow.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {
// ttnn.rpow(exponent, scalar_base) = pow(scalar_base, exponent)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_rpow(const std::uint32_t base_val) {
    sfpi::vFloat base_val_v = Converter::as_float(base_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _sfpu_binary_power_<is_fp32_dest_acc_en>(base_val_v, sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Op class for base ** x.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
struct Rpow : SfpuUnaryOp<Rpow<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t base_val) {
        calculate_rpow<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(base_val);
    }
    static inline __attribute__((always_inline)) void init_op() { sfpu_binary_pow_init<APPROXIMATION_MODE>(); }
};

}  // namespace ckernel::sfpu

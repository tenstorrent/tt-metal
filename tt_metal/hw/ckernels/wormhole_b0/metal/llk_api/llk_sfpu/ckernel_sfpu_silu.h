// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (APPROXIMATION_MODE) {
        // Fast path: use hardware LUT-based sigmoid approximation.
        // The LUT coefficients are pre-loaded into L-registers by silu_init.
        // lut() computes a 3-segment piecewise-linear sigmoid(x)-0.5 in a single
        // hardware instruction, handling sign internally.
        sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            // sigmoid(x) via hardware LUT + 0.5 bias
            sfpi::vFloat sig = sfpi::lut(x, l0, l1, l2) + 0.5f;

            sfpi::vFloat result = x * sig;

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }

        sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
        sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
        sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    } else {
        // Accurate path: sigmoid via exp(-x) + Newton-Raphson reciprocal.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            // silu(x) = x * sigmoid(x)
            sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

            // Round to bfloat16 if not in fp32 accumulation mode
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    if constexpr (APPROXIMATION_MODE) {
        // Load LUT coefficients for sigmoid approximation
        sigmoid_appx_init();
    } else {
        _init_sfpu_reciprocal_<false>();
    }
}

}  // namespace ckernel::sfpu

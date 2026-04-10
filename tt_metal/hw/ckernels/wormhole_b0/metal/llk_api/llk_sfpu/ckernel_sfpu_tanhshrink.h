// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel::sfpu {

// Tanhshrink(x) = x - tanh(x)
// Single-pass SFPU kernel that computes tanh in-register and subtracts
// from the original value, avoiding multi-tile composite decomposition.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanhshrink() {
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::vFloat tanh_v = sfpi::lut(v, l0, l1, l2);
            sfpi::dst_reg[0] = v - tanh_v;
            sfpi::dst_reg++;
        }

        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];

            sfpi::vFloat tanh_v;
            if constexpr (is_fp32_dest_acc_en) {
                tanh_v = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(v);
            } else {
                tanh_v = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(v);
                tanh_v = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(tanh_v, 0));
            }

            sfpi::dst_reg[0] = v - tanh_v;
            sfpi::dst_reg++;
        }
    }
}

}  // namespace ckernel::sfpu

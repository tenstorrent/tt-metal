// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh_lut() {
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        val = lut(val, l0, l1, l2);
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

// tanh[x] = (exp[2x] - 1) / (exp[2x] + 1)
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh_accurate() {
    // SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input = dst_reg[0];
        vFloat val = input;
        val = val * 2.0f;                 // 2x
        val = _sfpu_exp_21f_<true>(val);  // exp(2x)
        vFloat denom = val + 1.0f;        // exp(2x) + 1
        if constexpr (fp32_dest_acc_en) {
            denom = _sfpu_reciprocal_<2>(denom);
        } else {
            denom = _sfpu_reciprocal_<1>(denom);
            denom = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(denom, 0));
        }
        // denom = _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, legacy_compat>(ITERATIONS);
        val = val - 1.0f;   // exp(2x) - 1
        val = val * denom;  // (exp(2x) - 1) / (exp(2x) + 1)

        v_if(input > 3.5f) { val = 1.0f; }
        // v_elseif (input < -3.5f){
        //     val = -1.0f;
        // }
        v_elseif(input == 0.0f) { val = 0.0f; }
        v_endif;
        dst_reg[0] = val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool fast_and_approx, bool fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if (fast_and_approx) {
        calculate_tanh_lut<APPROXIMATION_MODE, fp32_dest_acc_en, ITERATIONS>();
    } else {
        calculate_tanh_accurate<APPROXIMATION_MODE, fp32_dest_acc_en, ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE, bool fast_and_approx>
inline void tanh_init() {
    if (fast_and_approx) {
        uint imm0;
        uint imm1;
        uint imm2;
        imm0 = 0x1DFF;  // 0.90625*x
        imm1 = 0x481A;  // 0.09375*x + 0.8125
        imm2 = 0xFF00;  // 1
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
        _sfpu_load_imm16_(2, imm2);
    } else {
        _init_reciprocal_<APPROXIMATION_MODE, false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

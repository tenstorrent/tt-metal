// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_common_includes.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(void (*func)()) {
    func();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(
    const uint param0 = 0, const uint param1 = 0, const uint param2 = 0, const uint param3 = 0, const uint param4 = 0, const uint param5 = 0) {
    _llk_math_eltwise_unary_sfpu_init_();

    switch (sfpu_op) {
        case SfpuType::tanh:
        case SfpuType::tanh_derivative:
             sfpu::_init_tanh_<APPROXIMATE>();
             break;
        case SfpuType::dropout:
            sfpu::_init_dropout_(param2);
            break;
        default:
            break;
    }
}

}

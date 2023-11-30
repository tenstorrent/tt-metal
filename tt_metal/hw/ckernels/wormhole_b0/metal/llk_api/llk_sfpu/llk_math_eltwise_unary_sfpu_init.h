// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_common_includes.h"

namespace ckernel {

inline void eltwise_unary_sfpu_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(void (*func)()) {
    eltwise_unary_sfpu_configure_addrmod();
    func();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(
    const uint param0 = 0, const uint param1 = 0, const uint param2 = 0, const uint param3 = 0, const uint param4 = 0, const uint param5 = 0) {
    _llk_math_eltwise_unary_sfpu_init_();

    switch (sfpu_op) {
        case SfpuType::reciprocal:
            sfpu::_init_reciprocal_<APPROXIMATE>();
            break;
        case SfpuType::exponential:
            sfpu::_init_exponential_<APPROXIMATE>();
            break;
        case SfpuType::log:
            sfpu::_init_log_<APPROXIMATE>();
            break;
        case SfpuType::sqrt:
            sfpu::_init_sqrt_<APPROXIMATE>();
            break;
        case SfpuType::tanh:
        case SfpuType::tanh_derivative:
            sfpu::_init_tanh_<APPROXIMATE>();
            break;
        case SfpuType::sigmoid:
            sfpu::_init_sigmoid_<APPROXIMATE>();
            break;
        case SfpuType::gelu_derivative:
            sfpu::_init_gelu_derivative_<APPROXIMATE>();
            break;
        case SfpuType::gelu:
            sfpu::_init_gelu_<APPROXIMATE>();
            break;
        case SfpuType::dropout:
            sfpu::_init_dropout_(param2);
            break;
        default:
            break;
    }
}

}

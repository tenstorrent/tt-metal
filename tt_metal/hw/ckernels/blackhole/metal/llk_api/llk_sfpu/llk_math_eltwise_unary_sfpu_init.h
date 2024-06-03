// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init() {
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(void (*func)()) {
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    func();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init_1_param(void (*func)(uint), uint param0 = 0) {
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    func(param0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}

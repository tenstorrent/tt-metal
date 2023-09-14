/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llk_math_eltwise_unary_sfpu_common_includes.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(void (*func)()) {
    func();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}

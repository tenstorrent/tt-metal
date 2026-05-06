// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp2.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_exp2() {
    _calculate_exp2_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>();
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    _init_exp2_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

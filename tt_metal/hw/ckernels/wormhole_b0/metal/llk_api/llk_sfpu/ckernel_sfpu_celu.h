// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_celu.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    _calculate_celu_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(param0, param1);
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_selu.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint32_t scale, uint32_t alpha) {
    _calculate_selu_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(scale, alpha);
}

}  // namespace ckernel::sfpu

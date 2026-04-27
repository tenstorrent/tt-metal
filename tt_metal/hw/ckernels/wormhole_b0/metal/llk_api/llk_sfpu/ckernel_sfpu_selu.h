// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_selu.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) {
    _calculate_selu_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(scale, alpha);
}

}  // namespace sfpu
}  // namespace ckernel

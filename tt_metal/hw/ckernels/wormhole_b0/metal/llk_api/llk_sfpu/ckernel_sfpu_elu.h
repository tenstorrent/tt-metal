// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_elu.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_elu(uint32_t dst_index_in, uint32_t dst_index_out, uint slope) {
    _calculate_elu_<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(dst_index_in, dst_index_out, slope);
}

}  // namespace sfpu
}  // namespace ckernel

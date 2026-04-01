// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
// NOTE: The following header is from the external tt-llk and not defined in tt-metal
#include "sfpu/ckernel_sfpu_square.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square(uint32_t dst_index_in, uint32_t dst_index_out) {
    _calculate_square_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

}  // namespace ckernel::sfpu

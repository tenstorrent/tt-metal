// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_reshuffle_rows.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_reshuffle_rows(uint32_t dst_index_in, uint32_t dst_index_out, uint idx_addr) {
    _calculate_reshuffle_rows_(dst_index_in, dst_index_out, idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel

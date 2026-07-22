// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

inline void my_add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

    for (size_t i = 0; i < 8; i++) {
        sfpi::vFloat a = sfpi::dst_reg[in0_base_idx + i];
        sfpi::vFloat b = sfpi::dst_reg[in1_base_idx + i];
        sfpi::dst_reg[out_base_idx + i] = a + b;
    }
}

}  // namespace ckernel::sfpu

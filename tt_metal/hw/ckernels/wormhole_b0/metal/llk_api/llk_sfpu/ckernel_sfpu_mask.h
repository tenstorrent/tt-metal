// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    const bool exponent_size_8 = true;
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        vFloat val = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val;
        v_if(_sfpu_is_fp16_zero_(mask, exponent_size_8)) { dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = vConst0; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_int_mask(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    const int mask_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt mask = dst_reg[mask_idx];
        vFloat val = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val;
        v_if(mask == 0) { dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = vConst0; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_posinf(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    const bool exponent_size_8 = true;
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        vFloat val = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val;
        v_if(_sfpu_is_fp16_zero_(mask, exponent_size_8)) {
            dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = std::numeric_limits<float>::infinity();
        }
        v_endif;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

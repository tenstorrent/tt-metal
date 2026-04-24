// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_relu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void relu_min(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint uint_threshold) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_max(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint uint_threshold) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(dst_index_in, dst_index_out, ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel

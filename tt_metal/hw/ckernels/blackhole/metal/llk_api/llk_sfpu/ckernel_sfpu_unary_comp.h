// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v == s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v == s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v > s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v < s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v < s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint value) {
    // SFPU microcode
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if(v > s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = v;

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

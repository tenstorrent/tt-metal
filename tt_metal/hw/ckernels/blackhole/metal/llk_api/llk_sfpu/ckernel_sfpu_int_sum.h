// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    for (size_t i = 0; i < 2; ++i) {
        vInt a = dst_reg[i];

        for (size_t j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j];
            a += b;
        }

        for (size_t j = 16; j < 24; j += 2) {
            vInt b = dst_reg[i + j];
            a += b;
        }

        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_row(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    for (size_t i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i];

        int arr[] = {1, 8, 9};
        for (size_t j = 0; j < sizeof(arr) / sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]];
            a += b;
        }

        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void sum_int_init() {}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void add_int(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint dst_offset) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        vInt b = dst_reg[dst_index_in * SFP_DST_TILE_ROWS + 32];

        vInt r = a + b;

        dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = r;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

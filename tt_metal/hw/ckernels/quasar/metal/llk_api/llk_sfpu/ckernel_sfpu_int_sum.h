// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

#include "sfpi.h"
#include <cstddef>

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
    for (size_t i = 0; i < 2; ++i) {
        vInt a = dst_reg[i].mode<DataLayout::SM32>();

        for (size_t j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j].mode<DataLayout::SM32>();
            a += b;
        }

        for (size_t j = 16; j < 24; j += 2) {
            vInt b = dst_reg[i + j].mode<DataLayout::SM32>();
            a += b;
        }

        dst_reg[i].mode<DataLayout::SM32>() = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_row() {
    for (size_t i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i].mode<DataLayout::SM32>();

        int arr[] = {1, 8, 9};
        for (size_t j = 0; j < sizeof(arr) / sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]].mode<DataLayout::SM32>();
            a += b;
        }

        dst_reg[i].mode<DataLayout::SM32>() = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void sum_int_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void add_int([[maybe_unused]] const uint dst_offset) {
    constexpr uint dst_tile_size_sfpi =
        (1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32)) / sfpi::SFP_DESTREG_STRIDE;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[0].mode<DataLayout::SM32>();
        vInt b = dst_reg[dst_tile_size_sfpi].mode<DataLayout::SM32>();

        vInt r = a + b;

        dst_reg[0].mode<DataLayout::SM32>() = r;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

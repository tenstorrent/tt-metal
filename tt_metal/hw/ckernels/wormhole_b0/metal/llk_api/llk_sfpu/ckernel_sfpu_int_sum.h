// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Dest holds the int32 tiles as two's complement (the unpacker copies the L1 words into a
// 32-bit Dest unchanged), which is what SFPIADD wants. Every access is therefore a raw
// DataLayout::I32 load / store: the default vInt layout on Wormhole (SM32, hardware mode 12)
// would convert sign-magnitude <-> two's complement on the way in and out, and the previous
// software helpers only existed to undo that conversion again. The raw path drops both
// conversions, and is the only one that keeps INT32_MIN (0x80000000), which the converting
// path reads and stores as 0.

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
    for (unsigned i = 0; i < 2; ++i) {
        vInt a = dst_reg[i].mode<sfpi::DataLayout::I32>();

        for (unsigned j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j].mode<sfpi::DataLayout::I32>();
            a += b;
        }

        for (unsigned j = 16; j < 24; j += 2) {
            vInt b = dst_reg[i + j].mode<sfpi::DataLayout::I32>();
            a += b;
        }

        dst_reg[i].mode<sfpi::DataLayout::I32>() = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_row() {
    for (unsigned i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i].mode<sfpi::DataLayout::I32>();

        int arr[] = {1, 8, 9};
        for (unsigned j = 0; j < sizeof(arr) / sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]].mode<sfpi::DataLayout::I32>();
            a += b;
        }

        dst_reg[i].mode<sfpi::DataLayout::I32>() = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void sum_int_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void add_int([[maybe_unused]] const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[0].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[32].mode<sfpi::DataLayout::I32>();

        dst_reg[0].mode<sfpi::DataLayout::I32>() = a + b;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

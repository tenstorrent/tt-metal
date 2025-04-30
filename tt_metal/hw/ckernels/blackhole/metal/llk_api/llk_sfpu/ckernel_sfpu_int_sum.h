// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
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
inline void calculate_sum_int_row() {
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
inline void add_int(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[0];
        vInt b = dst_reg[32];

        vInt r = a + b;

        dst_reg[0] = r;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

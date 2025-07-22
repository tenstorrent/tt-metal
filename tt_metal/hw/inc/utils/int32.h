// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

inline constexpr int32_t NEG_INF_INT32 = 0x80000000;  // Representation of minimum int32 value
inline constexpr int32_t POS_INF_INT32 = 0x7FFFFFFF;  // Representation of maximum int32 value

// Optimized function to compare two int32 values
bool int32_greater(int32_t int32_a, int32_t int32_b) {
    /*
    int32 format (32 bits total):
    [Sign (1 bit)][Value (31 bits)]
       bit 31         bits 30-0

    Comparison Logic:
    - If signs differ:
        - If int32_a is positive (sign bit 0), it is greater.
        - If int32_a is negative (sign bit 1), it is not greater.
    - If signs are the same:
        - Positive numbers: higher value means greater.
        - Negative numbers: higher value means greater (two's complement).
    */

    // Check if signs are different
    if ((int32_a ^ int32_b) & 0x80000000) {
        // Signs differ: if int32_a is positive, it's greater
        return (int32_a & 0x80000000) == 0;
    }

    // Signs are the same - simple comparison works for two's complement
    return int32_a > int32_b;
}

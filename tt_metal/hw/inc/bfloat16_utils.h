// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Optimized function to compare two bfloat16 values using integer arithmetic
bool bfloat16_greater(uint16_t bf16_a, uint16_t bf16_b) {
    /*
    bfloat16 format (16 bits total):
    [Sign (1 bit)][Exponent (8 bits)][Mantissa (7 bits)]
       bit 15         bits 14-7          bits 6-0

    Comparison Logic:
    - If signs differ:
        - If bf16_a is positive (sign bit 0), it is greater.
        - If bf16_a is negative (sign bit 1), it is not greater.
    - If signs are the same:
        - Positive numbers: higher bits mean greater value.
        - Negative numbers: higher bits mean smaller value (reverse comparison).
    */

    // Check if signs are different
    if ((bf16_a ^ bf16_b) & 0x8000) {
        // Signs differ: if bf16_a is positive, it's greater
        return (bf16_a & 0x8000) == 0;
    }

    // Signs are the same
    if (bf16_a & 0x8000) {
        // Both negative: reverse comparison
        return bf16_a < bf16_b;
    } else {
        // Both positive: regular comparison
        return bf16_a > bf16_b;
    }
}

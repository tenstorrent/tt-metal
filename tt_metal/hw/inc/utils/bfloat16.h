// SPDX-FileCopyrightText: © 2043 Tenstorrent Inc.
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

// Function to add two bfloat16 values using integer arithmetic
uint16_t bfloat16_add(uint16_t bf16_a, uint16_t bf16_b) {
    // Extract the sign, exponent, and mantissa from both values
    uint16_t sign_a = bf16_a & 0x8000;
    uint16_t sign_b = bf16_b & 0x8000;
    int16_t exp_a = (bf16_a & 0x7F80) >> 7;
    int16_t exp_b = (bf16_b & 0x7F80) >> 7;
    uint16_t mant_a = (bf16_a & 0x007F) | 0x0080;  // Add implicit leading 1
    uint16_t mant_b = (bf16_b & 0x007F) | 0x0080;  // Add implicit leading 1

    // Handle subnormal numbers (exponent is zero)
    if (exp_a == 0) {
        mant_a &= 0x007F;  // Remove implicit leading 1
    }
    if (exp_b == 0) {
        mant_b &= 0x007F;
    }

    // Align the mantissas by shifting the smaller one
    if (exp_a > exp_b) {
        mant_b >>= (exp_a - exp_b);
    } else if (exp_b > exp_a) {
        mant_a >>= (exp_b - exp_a);
        exp_a = exp_b;
    }

    // Add or subtract mantissas based on signs
    uint16_t mant_res;
    uint16_t sign_res;
    if (sign_a == sign_b) {
        mant_res = mant_a + mant_b;
        sign_res = sign_a;  // Result keeps the same sign
    } else {
        if (mant_a >= mant_b) {
            mant_res = mant_a - mant_b;
            sign_res = sign_a;  // Result keeps the sign of the larger magnitude
        } else {
            mant_res = mant_b - mant_a;
            sign_res = sign_b;
        }
    }

    // Normalize the result
    if (mant_res & 0x0100) {  // Mantissa overflow
        mant_res >>= 1;
        exp_a += 1;
    }
    while (mant_res && !(mant_res & 0x0080)) {  // Normalize mantissa (shift left)
        mant_res <<= 1;
        exp_a -= 1;
    }

    // Handle exponent overflow and underflow
    if (exp_a >= 0xFF) {  // Overflow to infinity
        return sign_res | 0x7F80;
    }
    if (exp_a <= 0) {  // Underflow to zero
        return sign_res;
    }

    // Combine the result
    uint16_t result = sign_res | (exp_a << 7) | (mant_res & 0x007F);
    return result;
}

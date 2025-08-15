// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

inline constexpr uint32_t NEG_INF_FLOAT32 = 0xFF800000;  // Representation of negative infinity in float32
inline constexpr uint32_t POS_INF_FLOAT32 = 0x7F800000;  // Representation of positive infinity in float32
inline constexpr uint32_t NAN_FLOAT32 = 0x7FFFFFFF;      // Representation of NaN in float32
inline constexpr uint32_t FLOAT32_SIGN_MASK = 0x80000000;       // Sign bit mask for float32
inline constexpr uint32_t FLOAT32_EXPONENT_MASK = 0x7F800000;   // Exponent mask for float32
inline constexpr uint32_t FLOAT32_MANTISSA_MASK = 0x007FFFFF;   // Mantissa mask for float32
inline constexpr uint32_t FLOAT32_MAGNITUDE_MASK = 0x7FFFFFFF;  // Magnitude mask (all bits except sign)

// Optimized function to compare two float32 values using integer arithmetic
bool float32_greater(uint32_t f32_a, uint32_t f32_b) {
    /*
    float32 format (32 bits total):
    [Sign (1 bit)][Exponent (8 bits)][Mantissa (23 bits)]
       bit 31         bits 30-23          bits 22-0

    Comparison Logic:
    - Handle special cases (NaN, infinity) first
    - If signs differ:
        - If f32_a is positive (sign bit 0), it is greater.
        - If f32_a is negative (sign bit 1), it is not greater.
    - If signs are the same:
        - Positive numbers: higher bits mean greater value.
        - Negative numbers: higher bits mean smaller value (reverse comparison).
    */

    // Handle NaN cases - NaN is never greater than anything
    if (((f32_a & FLOAT32_EXPONENT_MASK) == FLOAT32_EXPONENT_MASK && (f32_a & FLOAT32_MANTISSA_MASK) != 0) ||
        ((f32_b & FLOAT32_EXPONENT_MASK) == FLOAT32_EXPONENT_MASK && (f32_b & FLOAT32_MANTISSA_MASK) != 0)) {
        return false;
    }

    // Handle zero cases (both +0 and -0 are equal)
    if ((f32_a & FLOAT32_MAGNITUDE_MASK) == 0 && (f32_b & FLOAT32_MAGNITUDE_MASK) == 0) {
        return false;
    }

    // Check if signs are different
    if ((f32_a ^ f32_b) & FLOAT32_SIGN_MASK) {
        // Signs differ: if f32_a is positive, it's greater
        return (f32_a & FLOAT32_SIGN_MASK) == 0;
    }

    // Signs are the same
    if (f32_a & FLOAT32_SIGN_MASK) {
        // Both negative: reverse comparison
        return f32_a < f32_b;
    } else {
        // Both positive: regular comparison
        return f32_a > f32_b;
    }
}

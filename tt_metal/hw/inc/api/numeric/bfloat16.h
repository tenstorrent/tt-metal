// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>
#include <stdint.h>

#include "internal/risc_attribs.h"

inline constexpr uint16_t NEG_INF_BFLOAT16 = 0xFF80;    // Representation of negative infinity in bfloat16
inline constexpr uint16_t POS_INF_BFLOAT16 = 0x7F80;    // Representation of positive infinity in bfloat16
inline constexpr uint16_t NAN_BFLOAT16 = 0x7FFF;        // Representation of NaN in bfloat16
inline constexpr uint16_t BFLOAT16_SIGN_MASK = 0x8000;  // Sign bit mask for bfloat16

// Convert a single-precision float to bfloat16 using IEEE 754 round-to-nearest-even.
// Matches the packer hardware semantics, so values produced via this helper compare
// bit-identically against values rounded down to bf16 by the packer.
FORCE_INLINE std::uint16_t fp32_to_bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    std::uint32_t lsb = (bits >> 16) & 1u;
    std::uint32_t rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;

    return static_cast<std::uint16_t>(bits >> 16);
}

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
    if ((bf16_a ^ bf16_b) & BFLOAT16_SIGN_MASK) {
        // Signs differ: if bf16_a is positive, it's greater
        return (bf16_a & BFLOAT16_SIGN_MASK) == 0;
    }

    // Signs are the same
    if (bf16_a & BFLOAT16_SIGN_MASK) {
        // Both negative: reverse comparison
        return bf16_a < bf16_b;
    } else {
        // Both positive: regular comparison
        return bf16_a > bf16_b;
    }
}

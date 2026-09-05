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

// Convert bfloat16 (stored as uint16_t) to float.
// Bfloat16 occupies the high 16 bits of a float's bit representation; this
// zero-extends the value into the low 16 bits.
FORCE_INLINE float bf16_to_fp32(std::uint16_t bf16) {
    std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Convert a single-precision float to bfloat16 using IEEE 754 round-to-nearest.
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

// Convert a single-precision float to bfloat16 by truncation (round toward zero).
// Faster than fp32_to_bf16 but drops the low 16 mantissa bits without rounding.
// Suitable for intermediate computations where the small rounding error is acceptable.
FORCE_INLINE std::uint16_t fp32_to_bf16_truncate(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

// Split a float into its high and low 16-bit halves for SFPU register loads,
// which only accept 16-bit immediate values.
// high16 holds the bfloat16 representation; low16 holds the discarded mantissa bits.
struct FloatBits {
    std::uint16_t high16;
    std::uint16_t low16;

    explicit FloatBits(float value) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        high16 = static_cast<std::uint16_t>(bits >> 16);
        low16 = static_cast<std::uint16_t>(bits & 0xFFFFu);
    }
};

inline constexpr uint16_t BFLOAT16_EXPONENT_MASK = 0x7F80;   // Exponent mask for bfloat16
inline constexpr uint16_t BFLOAT16_MANTISSA_MASK = 0x007F;   // Mantissa mask for bfloat16
inline constexpr uint16_t BFLOAT16_MAGNITUDE_MASK = 0x7FFF;  // Magnitude mask for bfloat16

// Optimized function to compare two bfloat16 values using integer arithmetic
bool bfloat16_greater(uint16_t bf16_a, uint16_t bf16_b) {
    /*
    bfloat16 format (16 bits total):
    [Sign (1 bit)][Exponent (8 bits)][Mantissa (7 bits)]
       bit 15         bits 14-7          bits 6-0

    Comparison Logic:
    - Torch argmax semantics: NaN is greater than any non-NaN value.
      Two NaNs are equal (neither is strictly greater, preserving the first NaN index).
    - Handle zero cases (both +0 and -0 are equal).
    - If signs differ:
        - If bf16_a is positive (sign bit 0), it is greater.
        - If bf16_a is negative (sign bit 1), it is not greater.
    - If signs are the same:
        - Positive numbers: higher bits mean greater value.
        - Negative numbers: higher bits mean smaller value (reverse comparison).
    */

    bool is_nan_a = (bf16_a & BFLOAT16_EXPONENT_MASK) == BFLOAT16_EXPONENT_MASK && (bf16_a & BFLOAT16_MANTISSA_MASK) != 0;
    bool is_nan_b = (bf16_b & BFLOAT16_EXPONENT_MASK) == BFLOAT16_EXPONENT_MASK && (bf16_b & BFLOAT16_MANTISSA_MASK) != 0;
    if (is_nan_a || is_nan_b) {
        return is_nan_a && !is_nan_b;
    }

    // Handle zero cases (both +0 and -0 are equal)
    if ((bf16_a & BFLOAT16_MAGNITUDE_MASK) == 0 && (bf16_b & BFLOAT16_MAGNITUDE_MASK) == 0) {
        return false;
    }

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

inline bool bfloat16_equal(uint16_t bf16_a, uint16_t bf16_b) {
    if (bf16_a == bf16_b) {
        return true;
    }
    // ±0 equality
    if ((bf16_a & BFLOAT16_MAGNITUDE_MASK) == 0 && (bf16_b & BFLOAT16_MAGNITUDE_MASK) == 0) {
        return true;
    }
    // NaN equality (for tie-breaking/first-index consistency)
    bool is_nan_a = (bf16_a & BFLOAT16_EXPONENT_MASK) == BFLOAT16_EXPONENT_MASK && (bf16_a & BFLOAT16_MANTISSA_MASK) != 0;
    bool is_nan_b = (bf16_b & BFLOAT16_EXPONENT_MASK) == BFLOAT16_EXPONENT_MASK && (bf16_b & BFLOAT16_MANTISSA_MASK) != 0;
    return is_nan_a && is_nan_b;
}

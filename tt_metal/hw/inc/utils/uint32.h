// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

inline constexpr uint32_t MIN_UINT32 = 0x00000000;  // Representation of minimum uint32 value
inline constexpr uint32_t MAX_UINT32 = 0xFFFFFFFF;  // Representation of maximum uint32 value

// Optimized function to compare two uint32 values
bool uint32_greater(uint32_t uint32_a, uint32_t uint32_b) {
    /*
    uint32 format (32 bits total):
    [Value (32 bits)]
       bits 31-0

    Comparison Logic:
    - Unsigned integers are straightforward to compare
    - Higher bit pattern always means greater value
    - No sign bit complications
    */

    // Simple comparison for unsigned integers
    return uint32_a > uint32_b;
}

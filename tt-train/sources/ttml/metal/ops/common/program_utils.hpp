// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>

inline uint32_t get_block_size(uint32_t num_inner, const uint32_t max_block_size = 4U) {
    for (uint32_t block_size = max_block_size; block_size > 1U; block_size--) {
        if (num_inner % block_size == 0) {  // if num_inner is divisible by block_size - choose this block_size
            return block_size;
        }
    }
    return 1U;
}

inline uint32_t pack_two_bfloat16_to_uint32(float value) {
    uint32_t uint32_data = std::bit_cast<uint32_t>(value);
    uint32_t casted_uint16_data = uint32_data >> 16U;
    return casted_uint16_data | (casted_uint16_data << 16U);
}

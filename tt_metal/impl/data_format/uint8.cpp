// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/uint8.hpp>

#include <random>
#include <tt_stl/assert.hpp>

uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8) | (static_cast<uint32_t>(c) << 16) |
           (static_cast<uint32_t>(d) << 24);
}

std::vector<uint32_t> create_random_vector_of_uint8(size_t num_bytes, int seed) {
    TT_FATAL(num_bytes % 4 == 0, "num_bytes must be divisible by 4, got {}", num_bytes);
    std::mt19937 rng(seed);
    // Generate full uint32_t values directly (assumes num_bytes % 4 == 0)
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    size_t num_words = num_bytes / 4;
    std::vector<uint32_t> result(num_words);

    for (uint32_t& word : result) {
        word = dist(rng);
    }

    return result;
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/int8.hpp>

#include <random>
#include <tt_stl/assert.hpp>

uint32_t pack_four_int8_into_uint32(int8_t a, int8_t b, int8_t c, int8_t d) {
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

std::vector<uint32_t> create_random_vector_of_int8(size_t num_bytes, int seed) {
    TT_FATAL(num_bytes % 4 == 0, "num_bytes must be divisible by 4, got {}", num_bytes);
    std::mt19937 rng(seed);
    // Int8 uses sign-magnitude format with range -127 to +127 (no -128 since that's -0 in sign-magnitude).
    std::uniform_int_distribution<int> dist(-127, 127);

    size_t num_words = num_bytes / 4;
    std::vector<uint32_t> result(num_words);

    for (uint32_t& word : result) {
        int8_t a = static_cast<int8_t>(dist(rng));
        int8_t b = static_cast<int8_t>(dist(rng));
        int8_t c = static_cast<int8_t>(dist(rng));
        int8_t d = static_cast<int8_t>(dist(rng));
        word = pack_four_int8_into_uint32(a, b, c, d);
    }

    return result;
}

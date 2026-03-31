// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/uint8.hpp>

#include <random>

uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8) | (static_cast<uint32_t>(c) << 16) |
           (static_cast<uint32_t>(d) << 24);
}

std::vector<uint32_t> create_random_vector_of_uint8(size_t num_bytes, int min_val, int max_val, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min_val, max_val);

    // num_bytes uint8 elements, packed 4 per uint32
    std::vector<uint32_t> result(num_bytes / sizeof(uint32_t), 0);
    for (uint32_t& word : result) {
        uint8_t a = static_cast<uint8_t>(dist(rng));
        uint8_t b = static_cast<uint8_t>(dist(rng));
        uint8_t c = static_cast<uint8_t>(dist(rng));
        uint8_t d = static_cast<uint8_t>(dist(rng));
        word = pack_four_uint8_into_uint32(a, b, c, d);
    }
    return result;
}

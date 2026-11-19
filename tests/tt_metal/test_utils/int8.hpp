// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::test_utils {

// Pack four bytes into a word, with a in the least significant byte.
inline uint32_t pack_four_int8_into_uint32(int8_t a, int8_t b, int8_t c, int8_t d) {
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

// Generate num_bytes signed bytes in [-127, 127], packed four per word.
// num_bytes must be divisible by four.
inline std::vector<uint32_t> create_random_vector_of_int8(size_t num_bytes, int seed) {
    TT_FATAL(num_bytes % 4 == 0, "num_bytes must be divisible by 4, got {}", num_bytes);
    std::mt19937 rng(seed);
    // Int8 uses sign-magnitude, so -128 (0x80) would encode -0; the range is -127..127.
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

// Convert a sign-magnitude int8 byte (bit7=sign, bits[6:0]=magnitude) to a signed int.
inline int sign_mag_byte_to_int8(uint8_t byte) {
    const int mag = static_cast<int>(byte & 0x7F);
    const bool neg = (byte & 0x80) != 0;
    return neg ? -mag : mag;
}

// Convert a signed int32 to a sign-magnitude word (bit31=sign, bits[30:0]=magnitude).
inline uint32_t int32_to_sign_mag_word(int32_t value) {
    if (value >= 0) {
        return static_cast<uint32_t>(value);
    }
    return 0x80000000u | static_cast<uint32_t>(-static_cast<int64_t>(value));
}

}  // namespace tt::test_utils

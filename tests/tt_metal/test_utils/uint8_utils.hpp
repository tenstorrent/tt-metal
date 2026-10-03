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

// Returns num_bytes random uint8 values packed 4 per uint32. num_bytes must be divisible by 4.
inline std::vector<uint32_t> create_random_packed_uint8(size_t num_bytes, int seed) {
    TT_FATAL(num_bytes % 4 == 0, "num_bytes must be divisible by 4, got {}", num_bytes);
    std::mt19937 rng(seed);
    // Each uniformly drawn uint32 word supplies 4 independent uniform uint8 values.
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    std::vector<uint32_t> result(num_bytes / 4);
    for (uint32_t& word : result) {
        word = dist(rng);
    }
    return result;
}

}  // namespace tt::test_utils

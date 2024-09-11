// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    BufferType buftype;
};

struct BufferStressTestConfig {
    // Used for normal write/read tests
    uint32_t seed;
    uint32_t num_pages_total;

    uint32_t page_size;
    uint32_t max_num_pages_per_buffer;

    // Used for wrap test
    uint32_t num_iterations;
    uint32_t num_unique_vectors;
};


inline std::vector<uint32_t> generate_arange_vector(uint32_t size_bytes, uint32_t start = 0) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "Error");
    vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = start + i;
    }
    return src;
}

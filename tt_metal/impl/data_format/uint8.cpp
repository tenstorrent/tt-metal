// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/uint8.hpp>

uint32_t pack_uint8_into_uint32(uint8_t value) {
    uint32_t v = (uint32_t) value;
    return v | v << 8 | v << 16 | v << 24;
}

std::vector<std::uint32_t> create_constant_vector_of_uint8(uint32_t num_bytes, uint8_t value) {
    const uint32_t num_elements_vec = static_cast<uint32_t>(num_bytes / sizeof(std::uint32_t));
    std::vector<std::uint32_t> vec(num_elements_vec, 0);
    for (int i = 0; i < vec.size(); i++) {
        vec.at(i) = pack_uint8_into_uint32(value);
    }

    return vec;
}
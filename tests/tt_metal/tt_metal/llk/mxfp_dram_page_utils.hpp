// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace tt::tt_metal::unit_tests::llk::mxfp_typecast_utils {

inline uint32_t align_up(uint32_t value, uint32_t alignment) {
    return alignment ? ((value + alignment - 1) / alignment) * alignment : value;
}

inline std::vector<uint32_t> pad_dram_pages(
    const std::vector<uint32_t>& packed,
    uint32_t num_tiles,
    uint32_t tile_size,
    uint32_t dram_page_stride,
    uint32_t num_banks,
    uint32_t bank_id) {
    std::vector<uint32_t> padded(num_tiles * num_banks * dram_page_stride / sizeof(uint32_t), 0);
    auto* dst = reinterpret_cast<uint8_t*>(padded.data());
    const auto* src = reinterpret_cast<const uint8_t*>(packed.data());
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t page = tile * num_banks + bank_id;
        std::memcpy(dst + page * dram_page_stride, src + tile * tile_size, tile_size);
    }
    return padded;
}

inline std::vector<uint32_t> compact_dram_pages(
    const std::vector<uint32_t>& padded,
    uint32_t num_tiles,
    uint32_t tile_size,
    uint32_t dram_page_stride,
    uint32_t num_banks,
    uint32_t bank_id) {
    std::vector<uint32_t> packed(num_tiles * tile_size / sizeof(uint32_t), 0);
    auto* dst = reinterpret_cast<uint8_t*>(packed.data());
    const auto* src = reinterpret_cast<const uint8_t*>(padded.data());
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t page = tile * num_banks + bank_id;
        std::memcpy(dst + tile * tile_size, src + page * dram_page_stride, tile_size);
    }
    return packed;
}

}  // namespace tt::tt_metal::unit_tests::llk::mxfp_typecast_utils

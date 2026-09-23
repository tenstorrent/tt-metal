// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "zero_l1.hpp"
template <uint32_t Tiles> inline void compact_bf16_rows(uint32_t base) {
    auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    // Forward traversal is safe: each destination precedes all unread sources.
    for (uint32_t tile = 0; tile < Tiles; ++tile) {
        for (uint32_t word = 0; word < 8; ++word) {
            data[tile * 16 + word] = data[tile * 512 + word];
            data[tile * 16 + 8 + word] = data[tile * 512 + 128 + word];
        }
    }
}
template <uint32_t Bytes> inline void zero_compact_input(uint32_t base) {
    constexpr uint32_t prefix = []() constexpr { uint32_t n = 512; while (n * 2 <= Bytes) { n *= 2; } return n; }();
    zero_l1<prefix>(base);
    if constexpr (Bytes > prefix) {
        noc_async_read(get_noc_addr(base), base + prefix, Bytes - prefix);
        noc_async_read_barrier();
    }
}
template <uint32_t Tiles> inline void expand_bf16_rows(uint32_t source, uint32_t destination) {
    const auto* input = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source);
    auto* output = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination);
    for (uint32_t tile = 0; tile < Tiles; ++tile) {
        for (uint32_t word = 0; word < 8; ++word) {
            output[tile * 512 + word] = input[tile * 16 + word];
            output[tile * 512 + 128 + word] = input[tile * 16 + 8 + word];
        }
    }
}

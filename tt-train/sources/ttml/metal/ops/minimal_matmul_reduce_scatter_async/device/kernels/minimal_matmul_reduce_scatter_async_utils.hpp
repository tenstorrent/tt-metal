// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace mmrs {
struct Span {
    uint32_t start = 0;
    uint32_t size = 0;
};

constexpr Span balanced_span(uint32_t i, uint32_t total, uint32_t parts) {
    const uint32_t base = total / parts;
    const uint32_t rem = total % parts;
    return {(i * base) + (i < rem ? i : rem), base + (i < rem ? 1u : 0u)};
}

constexpr Span clip(Span block, uint32_t share_width) {
    if (block.start >= share_width) {
        return {block.start, 0};
    }
    const uint32_t left = share_width - block.start;
    return {block.start, block.size < left ? block.size : left};
}

constexpr Span n_block(uint32_t nb, uint32_t width, uint32_t N_blocks) {
    return balanced_span(nb, width, N_blocks);
}

constexpr uint32_t step_slice(uint32_t step, uint32_t ring_index, uint32_t ring_size) {
    return (ring_index + (2 * ring_size) - 1 - step) % ring_size;
}

template <bool transpose_a>
inline uint32_t input_page(uint32_t m, uint32_t k, uint32_t M_tiles, uint32_t K_tiles) {
    if constexpr (transpose_a) {
        return (k * M_tiles) + m;
    } else {
        return (m * K_tiles) + k;
    }
}

template <bool transpose_b>
inline uint32_t weight_page(uint32_t n, uint32_t k, uint32_t K_tiles, uint32_t N_tiles) {
    if constexpr (transpose_b) {
        return (n * K_tiles) + k;
    } else {
        return (k * N_tiles) + n;
    }
}
}  // namespace mmrs

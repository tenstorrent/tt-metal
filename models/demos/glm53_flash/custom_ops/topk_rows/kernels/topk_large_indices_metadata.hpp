// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Model-local copy (custom_ops/topk_rows): the reader -> compute mailbox carries a third word, the PHANTOM chunks of
// the row (the live-bound segment mode, custom_ops/sparse_bound.py): trailing all--inf chunks of the unbounded row that
// the reader pushes no page for and the classic compute body replays as merge + rebuild only.

#pragma once

#include <cstddef>
#include <cstdint>

struct TopkMetadataBounds {
    uint32_t num_chunks;
    uint32_t tail_elements;
    uint32_t phantom_chunks;
};

constexpr TopkMetadataBounds calculate_topk_bounds(uint32_t search_len, uint32_t chunk_elements) {
    const uint32_t num_chunks = (search_len + chunk_elements - 1) / chunk_elements;
    return {
        .num_chunks = num_chunks,
        .tail_elements = search_len - ((num_chunks - 1) * chunk_elements),
        .phantom_chunks = 0,
    };
}

inline constexpr uint32_t topk_metadata_num_chunks_word = offsetof(TopkMetadataBounds, num_chunks) / sizeof(uint32_t);
inline constexpr uint32_t topk_metadata_tail_elements_word =
    offsetof(TopkMetadataBounds, tail_elements) / sizeof(uint32_t);
inline constexpr uint32_t topk_metadata_phantom_chunks_word =
    offsetof(TopkMetadataBounds, phantom_chunks) / sizeof(uint32_t);

static_assert(sizeof(TopkMetadataBounds) == 3 * sizeof(uint32_t));

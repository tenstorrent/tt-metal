// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Work item arithmetic shared by the host (work split, cost model) and the three kernels. A work item
// is Sq_chunk_t query tile-rows of heads_per_item consecutive Q heads (one K/V group):
//   item = (batch * stacks_per_batch + stack) * q_num_chunks + chunk,  stacks_per_batch = q_heads / heads_per_item
// Plain C++17, no kernel or host includes.
namespace sdpa_fw_items {

constexpr uint32_t first_row_tile_of(uint32_t item, uint32_t q_num_chunks, uint32_t Sq_chunk_t) {
    return (item % q_num_chunks) * Sq_chunk_t;
}

// K/V steps (Sk_chunk_t tiles each) a chunk attends to: causal up to its last diagonal tile, else all.
constexpr uint32_t num_kv_steps_of_chunk(
    bool causal, uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t first_row_tile) {
    return causal ? (first_row_tile + Sq_chunk_t + Sk_chunk_t - 1U) / Sk_chunk_t : Ht / Sk_chunk_t;
}

struct ItemCoords {
    uint32_t batch;
    uint32_t head0;  // first Q head of the stack
    uint32_t first_row_tile;
};

constexpr ItemCoords decode_item(
    uint32_t item, uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item) {
    const uint32_t stacks_per_batch = q_heads / heads_per_item;
    const uint32_t stack = item / q_num_chunks;
    return {
        stack / stacks_per_batch,
        (stack % stacks_per_batch) * heads_per_item,
        first_row_tile_of(item, q_num_chunks, Sq_chunk_t)};
}

}  // namespace sdpa_fw_items

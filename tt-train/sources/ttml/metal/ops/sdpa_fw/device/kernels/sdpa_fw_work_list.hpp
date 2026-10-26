// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Work item = Sq_chunk_t query tile-rows of heads_per_item consecutive Q heads (one K/V group),
// item = (batch * stacks_per_batch + stack) * q_num_chunks + chunk; row r of an item is head
// head0 + r / Sq_chunk_t, tile row first_row_tile + r % Sq_chunk_t. Runtime args after the kernel's own:
// num_runs, then (first_item, num_items) per run. All three kernels walk the same list.
template <typename Fn>
inline void for_each_work_item(uint32_t arg_idx, Fn&& fn) {
    const uint32_t num_runs = get_arg_val<uint32_t>(arg_idx++);
    for (uint32_t run = 0; run < num_runs; ++run) {
        const uint32_t first_item = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t num_items = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t item = first_item; item < first_item + num_items; ++item) {
            fn(item);
        }
    }
}

struct WorkItemCoords {
    uint32_t batch;
    uint32_t head0;
    uint32_t first_row_tile;
};

template <uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item>
inline WorkItemCoords decode_work_item(uint32_t item) {
    constexpr uint32_t stacks_per_batch = q_heads / heads_per_item;
    const uint32_t stack = item / q_num_chunks;
    return {stack / stacks_per_batch, (stack % stacks_per_batch) * heads_per_item, (item % q_num_chunks) * Sq_chunk_t};
}

// K/V steps of Sk_chunk_t tiles a chunk attends to: causal up to its last diagonal tile, else all.
template <uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t>
inline uint32_t num_kv_steps_of_chunk([[maybe_unused]] uint32_t first_row_tile) {
#ifdef CAUSAL_MASK
    return (first_row_tile + Sq_chunk_t + Sk_chunk_t - 1U) / Sk_chunk_t;
#else
    return Ht / Sk_chunk_t;
#endif
}

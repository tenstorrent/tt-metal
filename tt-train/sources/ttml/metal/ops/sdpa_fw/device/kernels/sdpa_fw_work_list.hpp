// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The per-core work list, shared by the reader, writer and compute kernels.
//
// A work item is one Q chunk of one head stack: Sq_chunk_t query tile-rows of heads_per_item
// consecutive Q heads that share a K/V head. Stack s of a batch holds heads [s * heads_per_item,
// (s + 1) * heads_per_item); heads_per_item divides the group size, so a stack never straddles two
// K/V groups. item = (batch * stacks_per_batch + stack) * q_num_chunks + chunk. In the kernels the
// item's rows are stacked head after head: row r is head head0 + r / Sq_chunk_t, tile row
// first_row_tile + r % Sq_chunk_t.
//
// The list sits in the runtime args after the kernel's own arguments as num_runs, then
// (first_item, num_items) per run of consecutive items. All three kernels must walk it identically.
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
    uint32_t head0;           // first Q head of the stack
    uint32_t first_row_tile;  // first query tile-row of the chunk
};

template <uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item>
inline WorkItemCoords decode_work_item(uint32_t item) {
    constexpr uint32_t stacks_per_batch = q_heads / heads_per_item;
    const uint32_t stack = item / q_num_chunks;  // batch * stacks_per_batch + stack
    return {
        /* batch */ stack / stacks_per_batch,
        /* head0 */ (stack % stacks_per_batch) * heads_per_item,
        /* first_row_tile */ (item % q_num_chunks) * Sq_chunk_t};
}

// K/V steps (Sk_chunk_t tiles each) a Q chunk attends to. Causal: up to the step holding the chunk's
// last diagonal tile; otherwise the whole sequence.
template <uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t>
inline uint32_t num_kv_steps_of_chunk([[maybe_unused]] uint32_t first_row_tile) {
#ifdef CAUSAL_MASK
    return (first_row_tile + Sq_chunk_t + Sk_chunk_t - 1U) / Sk_chunk_t;
#else
    return Ht / Sk_chunk_t;
#endif
}

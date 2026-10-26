// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The per-core work list, shared by the reader, writer and compute kernels. A work item is one Q
// chunk: Sq_chunk_t query tile-rows of one batch/head, item = (batch * q_heads + head) * q_num_chunks
// + chunk. The list sits in the runtime args after the kernel's own arguments as num_runs, then
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

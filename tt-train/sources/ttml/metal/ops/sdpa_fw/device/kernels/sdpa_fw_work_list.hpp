// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_item_math.hpp"

// The core's work list sits in the runtime args after the kernel's own arguments: num_runs, then
// (first_item, num_items) per run of consecutive items (sdpa_fw_item_math.hpp). All three kernels walk
// the same list; row r of an item is head head0 + r / Sq_chunk_t, tile row first_row_tile + r % Sq_chunk_t.
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

using WorkItemCoords = sdpa_fw_items::ItemCoords;

template <uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item>
inline WorkItemCoords decode_work_item(uint32_t item) {
    return sdpa_fw_items::decode_item(item, q_num_chunks, Sq_chunk_t, q_heads, heads_per_item);
}

template <uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t>
inline uint32_t num_kv_steps_of_chunk(uint32_t first_row_tile) {
#ifdef CAUSAL_MASK
    constexpr bool causal = true;
#else
    constexpr bool causal = false;
#endif
    return sdpa_fw_items::num_kv_steps_of_chunk(causal, Ht, Sq_chunk_t, Sk_chunk_t, first_row_tile);
}

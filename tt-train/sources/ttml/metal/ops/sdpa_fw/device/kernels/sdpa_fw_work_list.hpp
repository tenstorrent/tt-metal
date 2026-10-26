// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_item_math.hpp"

// The core's work list sits in the runtime args after the kernel's own arguments: num_units, then three
// words per unit (sdpa_fw_config.cpp). Items are per sdpa_fw_item_math.hpp; row r of an item is head
// head0 + r / Sq_chunk_t, tile row first_row_tile + r % Sq_chunk_t. Unit kinds:
//   whole:  a run of consecutive items, each over its full K/V range;
//   sender: K/V steps [step_begin, step_end) of one item, result sent to a merger core as a partial;
//   merger: its own step range of one item, then the senders' partials folded in, then finalized.
// All three kernels walk the same list.
constexpr uint32_t kUnitWhole = 0U;
constexpr uint32_t kUnitSender = 1U;
constexpr uint32_t kUnitMerger = 2U;

struct WorkUnit {
    uint32_t item;
    uint32_t role;
    uint32_t num_items;     // whole
    uint32_t step_begin;    // sender / merger
    uint32_t step_end;      // sender / merger
    uint32_t num_partials;  // merger
    uint32_t sem_idx;       // merger: which of its core's semaphores the senders bump
    uint32_t slot;          // sender: scratch slot it writes; merger: first slot it reads
    uint32_t merger_x;      // sender: merger core, NoC coordinates
    uint32_t merger_y;
};

template <typename Fn>
inline void for_each_work_unit(uint32_t arg_idx, Fn&& fn) {
    const uint32_t num_units = get_arg_val<uint32_t>(arg_idx++);
    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t w0 = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t w1 = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t w2 = get_arg_val<uint32_t>(arg_idx++);
        WorkUnit unit{};
        unit.item = w0 & 0xFFFFFU;
        unit.role = (w0 >> 20U) & 0x3U;
        unit.num_partials = (w0 >> 22U) & 0x3FU;
        unit.sem_idx = (w0 >> 28U) & 0xFU;
        if (unit.role == kUnitWhole) {
            unit.num_items = w1;
        } else {
            unit.num_items = 1U;
            unit.step_begin = w1 & 0xFFFFU;
            unit.step_end = w1 >> 16U;
        }
        unit.slot = w2 & 0xFFFFU;
        unit.merger_x = (w2 >> 16U) & 0xFFU;
        unit.merger_y = (w2 >> 24U) & 0xFFU;
        fn(unit);
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

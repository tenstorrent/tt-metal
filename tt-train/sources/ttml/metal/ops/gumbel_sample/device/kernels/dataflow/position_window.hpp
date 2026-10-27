// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-entry position list, staged as a local window. Op-local to gumbel_sample (shared by its
// reader and writer); move to metal/common/dataflow_utils.hpp if a second op grows a use for it.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// A staged window of a [B, 1, 1, 1] token-position list. Reader and writer consume the SAME staged
// layout and the SAME clamp -- one takes clamped >> 5 (the tile row), the other clamped & 31 (the
// row within it) -- so staging, slot addressing and the clamp are single-sourced here. Only the
// entry window the core's contiguous tile run touches is staged, keeping L1 bytes and DRAM page
// reads proportional to the core's share of the batch rather than the global batch.
struct PositionWindow {
    uint32_t l1_base{};
    uint32_t slot_bytes{};
    uint32_t first_entry{};

    // Clamp BEFORE the caller splits the value into bit fields, so every consumer derives its
    // field from the same clamped value. Positions live in device memory (the host cannot
    // range-check them on the dispatch path): unclamped, a position in the tile-padding band
    // [logical_tokens, Ht*32) selects a zero-filled padding row -- silently wrong samples -- and a
    // position past Ht*32 reads outside the buffer. Clamping to the last real token also gives the
    // intended row for the classic off-by-one (position == prompt length). The ASSERT makes a bad
    // position loud under watcher.
    inline uint32_t clamped_position(uint32_t entry, uint32_t logical_tokens) const {
        // volatile so the load cannot be hoisted above stage_position_window's read barrier.
        const uint32_t position =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + (entry - first_entry) * slot_bytes);
        ASSERT(position < logical_tokens);
        return (position < logical_tokens) ? position : (logical_tokens - 1U);
    }
};

// Derives the window from the core's contiguous tile run in the position-mode virtual tile space
// (Wt tiles per entry): the run touches exactly start_tile / Wt ..= (start_tile + num_tiles - 1) / Wt.
// Derived here, once, so the two consuming kernels cannot compute different windows.
template <typename AddressGenerator>
inline PositionWindow stage_position_window(
    uint32_t cb_idx, const AddressGenerator& address_generator, uint32_t start_tile, uint32_t num_tiles, uint32_t Wt) {
    PositionWindow window;
    window.slot_bytes = address_generator.get_aligned_page_size();
    window.l1_base = get_write_ptr(cb_idx);
    window.first_entry = start_tile / Wt;
    const uint32_t last_entry = (start_tile + num_tiles - 1U) / Wt;
    uint32_t l1_addr = window.l1_base;
    for (uint32_t e = window.first_entry; e <= last_entry; ++e) {
        noc_async_read_page(e, address_generator, l1_addr);
        l1_addr += window.slot_bytes;
    }
    noc_async_read_barrier();
    return window;
}

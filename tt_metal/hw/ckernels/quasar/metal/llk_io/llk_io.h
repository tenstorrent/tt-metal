// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"

#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK))
// Cursor helpers shared by pack (wr_entry_idx) and unpack (rd_entry_idx).

// On a BLOCKED ring one op must cover exactly one block.
inline bool dfb_op_is_whole_share(const LocalDFBInterface& intf, std::uint32_t num_tiles) {
    return intf.block_size <= 1 || num_tiles * intf.stride_size_tiles == static_cast<std::uint32_t>(intf.block_size);
}

// Move one counter's cursor past an n-tile op: to the op's last entry, then `jump` entries to
// this counter's next entry. Wraps to the ring base at the end.
inline void dfb_step_slot(const LocalDFBInterface& intf, DFBTCSlot& slot, std::uint32_t n) {
#if defined(UCK_CHLKC_PACK)
    std::uint32_t entry_idx = slot.wr_entry_idx;
#else
    std::uint32_t entry_idx = slot.rd_entry_idx;
#endif
    entry_idx += (n - 1u) * intf.stride_size_tiles + intf.jump;
    if (dfb_slot_cursor_offset_units(intf, slot, entry_idx) >= slot.ring_size) {
        entry_idx = slot.base_entry_idx;
    }
#if defined(UCK_CHLKC_PACK)
    slot.wr_entry_idx = static_cast<std::uint16_t>(entry_idx);
#else
    slot.rd_entry_idx = static_cast<std::uint16_t>(entry_idx);
#endif
}

// Normal op: step the current counter's cursor, then move on to the next counter.
inline void dfb_advance_slot(LocalDFBInterface& intf, DFBTCSlot& slot, std::uint32_t num_tiles) {
    dfb_step_slot(intf, slot, num_tiles);
    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

// Split op: the block belongs to every counter (per_tc tiles each), so step every cursor and
// stay on counter 0.
inline void dfb_advance_all_slots(LocalDFBInterface& intf, std::uint32_t per_tc) {
    for (std::uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb_step_slot(intf, intf.tc_slots[i], per_tc);
    }
}

#endif

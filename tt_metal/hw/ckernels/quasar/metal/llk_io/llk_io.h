// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"

#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK))
// DFB cursor helpers shared by pack (moves wr_entry_idx) and unpack (moves rd_entry_idx).

// A raw LLK op on a BLOCKED ring (block_size > 1) must move this hart's whole share: num_tiles
// entries, stride_size_tiles apart, cover exactly one block. A plain ring takes any count.
inline bool dfb_op_is_whole_share(const LocalDFBInterface& intf, std::uint32_t num_tiles) {
    return intf.block_size <= 1 || num_tiles * intf.stride_size_tiles == static_cast<std::uint32_t>(intf.block_size);
}

// Each tile counter keeps its own cursor, a bookmark into just its entries of the ring. Move one
// counter's bookmark past an n-tile op of its own: n - 1 strides to the op's last entry, then
// `jump` entries to this counter's next entry (past the other counters'); a bookmark that runs
// off the end of the ring wraps to its base.
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

// One op is always this hart's whole share of a counter visit (asserted by the caller), so every
// op hands off: step this counter's bookmark past the op and rotate to the next counter, whose
// bookmark is already waiting exactly where the stream continues.
inline void dfb_advance_slot(LocalDFBInterface& intf, DFBTCSlot& slot, std::uint32_t num_tiles) {
    dfb_step_slot(intf, slot, num_tiles);
    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

// Split op (split_tc): the whole block this hart just packed / unpacked belongs to every one of
// its counters, per_tc tiles each, so every slot's bookmark steps past a per_tc-tile op of its own
// (the host puts the hop to this hart's next block in `jump` for split harts too). tc_idx stays
// put, so the next op still starts from slot 0.
inline void dfb_advance_all_slots(LocalDFBInterface& intf, std::uint32_t per_tc) {
    for (std::uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb_step_slot(intf, intf.tc_slots[i], per_tc);
    }
}

#endif

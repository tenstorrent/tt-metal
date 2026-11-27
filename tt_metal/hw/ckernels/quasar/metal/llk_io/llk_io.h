// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "llk_assert.h"

// The posted / acked tile counts are 16 bits wide; mask off the rest of the word when reading them.
constexpr std::uint32_t DFB_TILE_COUNTER_COUNT_MASK = 0xFFFFu;

// Unpack and pack only work on a DFB this thread itself consumes / produces; on any other DFB the cursor
// never moves and nothing waits for tiles.
#define LLK_DFB_ASSERT_MY_CONSUMER(intf, op) \
    LLK_ASSERT(((intf).tensix_trisc_mask & DFB_TRISC_MY_CONSUMER_ROLE) != 0, op ": this TRISC does not consume the DFB")
#define LLK_DFB_ASSERT_MY_PRODUCER(intf, op) \
    LLK_ASSERT(((intf).tensix_trisc_mask & DFB_TRISC_MY_PRODUCER_ROLE) != 0, op ": this TRISC does not produce the DFB")

#if DFB_TRISC_HAS_INTERFACE
// Whether an op on this side is split over all of this hart's tile counters (one whole-block op
// shared by every counter): a BLOCKED producer feeding STRIDED consumers, or a BLOCKED consumer fed
// by STRIDED producers.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap, bool IsProducer>
inline bool dfb_op_is_split(const LocalDFBInterface& intf) {
    if constexpr (Pap == dfb::AccessPattern::UNKNOWN || Cap == dfb::AccessPattern::UNKNOWN) {
        return intf.split_tc != 0;
    } else if constexpr (IsProducer) {
        return Pap == dfb::AccessPattern::BLOCKED && Cap == dfb::AccessPattern::STRIDED;
    } else {
        return Pap == dfb::AccessPattern::STRIDED && Cap == dfb::AccessPattern::BLOCKED;
    }
}

// Cursor helpers shared by the producer side (push moves wr_entry_idx) and the consumer side (pop
// moves rd_entry_idx).

// On a BLOCKED ring one op must cover exactly one block.
inline bool dfb_op_is_whole_share(const LocalDFBInterface& intf, std::uint32_t num_tiles) {
    return intf.block_size <= 1 || num_tiles * intf.stride_size_tiles == static_cast<std::uint32_t>(intf.block_size);
}

// Move one counter's cursor past an n-tile op: to the op's last entry, then `jump` entries to
// this counter's next entry. Wraps to the ring base at the end.
template <bool IsProducer>
inline void dfb_step_slot(const LocalDFBInterface& intf, DFBTCSlot& slot, std::uint32_t n) {
    std::uint32_t entry_idx = IsProducer ? slot.wr_entry_idx : slot.rd_entry_idx;
    entry_idx += (n - 1u) * intf.stride_size_tiles + intf.jump;
    if (dfb_slot_cursor_offset_units(intf, slot, entry_idx) >= slot.ring_size) {
        entry_idx = slot.base_entry_idx;
    }
    if constexpr (IsProducer) {
        slot.wr_entry_idx = static_cast<std::uint16_t>(entry_idx);
    } else {
        slot.rd_entry_idx = static_cast<std::uint16_t>(entry_idx);
    }
}

// Normal op: step the current counter's cursor, then move on to the next counter.
template <bool IsProducer>
inline void dfb_advance_slot(LocalDFBInterface& intf, DFBTCSlot& slot, std::uint32_t num_tiles) {
    dfb_step_slot<IsProducer>(intf, slot, num_tiles);
    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

// Split op: the block belongs to every counter (per_tc tiles each), so step every cursor and
// stay on counter 0.
template <bool IsProducer>
inline void dfb_advance_all_slots(LocalDFBInterface& intf, std::uint32_t per_tc) {
    for (std::uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb_step_slot<IsProducer>(intf, intf.tc_slots[i], per_tc);
    }
}

#endif

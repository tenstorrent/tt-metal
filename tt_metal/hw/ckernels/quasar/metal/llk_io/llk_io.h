// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"

#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK))
// Used by pack and unpack to advance DFB rd/wr tile offsets and the tile counter pointer
//
// Pack updates wr_entry_idx / wr_offset
// Unpack updates rd_entry_idx / rd_offset
inline void dfb_advance_slot(
    LocalDFBInterface& intf,
    DFBTCSlot& slot,
    std::uint32_t num_tiles) {
    const std::uint32_t num_words = num_tiles * intf.stride_size;

    std::uint32_t entry_idx;
    std::uint32_t offset;
#if defined(UCK_CHLKC_PACK)
    entry_idx = slot.wr_entry_idx;
    offset = slot.wr_offset;
#elif defined(UCK_CHLKC_UNPACK)
    entry_idx = slot.rd_entry_idx;
    offset = slot.rd_offset;
#endif

    entry_idx += num_tiles * static_cast<std::uint32_t>(intf.stride_size_tiles);
    offset += num_words;
    if (offset >= slot.ring_size) {
        offset = 0;
        entry_idx = slot.base_entry_idx;
    }

#if defined(UCK_CHLKC_PACK)
    slot.wr_entry_idx = static_cast<std::uint16_t>(entry_idx);
    slot.wr_offset = static_cast<std::uint16_t>(offset);
#elif defined(UCK_CHLKC_UNPACK)
    slot.rd_entry_idx = static_cast<std::uint16_t>(entry_idx);
    slot.rd_offset = static_cast<std::uint16_t>(offset);
#endif

    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

#endif

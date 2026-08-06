// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tools/profiler/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "llk_io.h"

/**
 * @brief  Wait for num_tiles of free space in the dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles of free space to wait for in dataflow buffer
 */
inline void llk_wait_for_free_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    uint32_t tc_id = dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
    TT_WAIT_FREE(ckernel::p_stall::STALL_PACK, num_tiles, tc_id);
}

/**
 * @brief  Push num_tiles into the dataflow buffer, increment write pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to push into dataflow buffer
 */
// Push N tiles to stream buffer (increment write pointer)
template <std::uint8_t PACK_SEL = 0x1>
inline void llk_push_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    auto& slot = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx];
    uint32_t tc_id = dfb::get_counter_id(slot.packed_tile_counter);
    // Update the tile counters values
    TT_PUSH_TILES(PACK_SEL, num_tiles, tc_id);
    // [TEN-4746 / #48552] Fence the PACK-side TDMA tile-counter mutation before any following
    // counter op (WAIT_FREE / reserve / wait_front) on the same tc_id can read it. PUSH_TILES is a
    // TDMA-class op executed on PACK0; a back-to-back SYNC-class counter read reaches the tile-counter
    // engine before PUSH_TILES retires and latches a wrong-Neo counter address (observed as a
    // TILE_COUNTERS index 0x10000 fault on the K-spill matmul path via cb_intermed0). RISC-core delay
    // alone does NOT fix this; a real backend stall until PACK0/THCON are idle is required.
    TTI_STALLWAIT(
        ckernel::p_stall::STALL_TDMA | ckernel::p_stall::STALL_SYNC,
        0,
        ckernel::p_stall::THCON,
        ckernel::p_stall::PACK0);

    local_dfb_interface.wr_entry_ptr = 0;
    dfb_advance_slot(local_dfb_interface, slot, num_tiles);
}

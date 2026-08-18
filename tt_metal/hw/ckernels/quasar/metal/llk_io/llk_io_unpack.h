// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "llk_io.h"

/**
 * @brief  Wait for num_tiles available in the incoming dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
inline void llk_wait_tiles(const std::int32_t dfb_id, const std::uint32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    std::uint32_t tc_id =
        dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, tc_id);
    // TEN-4746: arm this dfb; a real unpack (UNPACR) on it must clear this before the matching pop.
    LLK_TDMA_GUARD_NOTE_WAIT(dfb_id);
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
template <std::uint8_t UNPACK_SEL = 0x3>
inline void llk_pop_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    // TEN-4746: popping a dfb that was waited but never unpacked (no UNPACR since wait_tiles) is a HW
    // hazard -- the wait can resolve before tiles are available.
    LLK_TDMA_GUARD_ASSERT_DISARMED(
        dfb_id, "TEN-4746: llk_pop_tiles on a dfb with no unpack (UNPACR) since llk_wait_tiles");
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    auto& slot = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx];
    std::uint32_t tc_id = dfb::get_counter_id(slot.packed_tile_counter);

    // Wait until selected unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, tc_id);

    dfb_advance_slot(local_dfb_interface, slot, num_tiles);
}

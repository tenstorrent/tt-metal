// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "llk_io.h"

// Packers that must finish before PUSH_TILES posts the credit; a thread that drives no packer waits for none.
#if defined(UCK_CHLKC_ISOLATE_SFPU)
constexpr std::uint8_t DFB_PACK_DONE_WAIT_MASK = 0x0;
#else
constexpr std::uint8_t DFB_PACK_DONE_WAIT_MASK = 0x1;
#endif

// Wait for num_tiles of free space on counter tc_id: WAIT_FREE when a packer follows, a RISC poll when none does.
inline void dfb_trisc_wait_for_free(const std::int32_t num_tiles, const std::uint32_t tc_id) {
#if defined(UCK_CHLKC_ISOLATE_SFPU)
    ckernel::tensix_sync();  // let this thread's earlier PUSH_TILES finish before reading the counter
    while ((ckernel::trisc::tile_counters[tc_id].f.acked & 0xFFFFu) < static_cast<std::uint32_t>(num_tiles)) {
    }
#else
    TT_WAIT_FREE(ckernel::p_stall::STALL_PACK, num_tiles, tc_id);
#endif
}

/**
 * @brief  Wait for num_tiles of free space in the dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles of free space to wait for in dataflow buffer
 */
template <dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN, dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
inline void llk_wait_for_free_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    LLK_ASSERT(
        dfb_op_is_whole_share(local_dfb_interface, num_tiles),
        "llk_wait_for_free_tiles: an op on a BLOCKED ring must move this hart's whole share");
    if (dfb_op_is_split<Pap, Cap, true>(local_dfb_interface)) {
        // Split: the block will belong to every counter, wait for each one's share of free space.
        const std::int32_t per_tc = num_tiles / local_dfb_interface.num_tcs_to_rr;
        for (std::uint8_t i = 0; i < local_dfb_interface.num_tcs_to_rr; i++) {
            dfb_trisc_wait_for_free(per_tc, dfb::get_counter_id(local_dfb_interface.tc_slots[i].packed_tile_counter));
        }
    } else {
        std::uint32_t tc_id =
            dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
        dfb_trisc_wait_for_free(num_tiles, tc_id);
    }
    // The RISC poll issues no WAIT_FREE, so there is no guard to arm.
#if !defined(UCK_CHLKC_ISOLATE_SFPU)
    // TEN-4746: arm this dfb; a real pack (PACR) on it must clear this before the matching push.
    LLK_TDMA_GUARD_NOTE_WAIT(dfb_id);
#endif
}

/**
 * @brief  Push num_tiles into the dataflow buffer, increment write pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to push into dataflow buffer
 */
// Push N tiles to stream buffer (increment write pointer)
template <
    std::uint8_t PACK_SEL = 0x1,
    dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN,
    dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
inline void llk_push_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    // TEN-4746: pushing a dfb that was reserved but never packed (no PACR since wait_for_free) is a HW
    // hazard -- the free-space wait can resolve before space is actually available.
    LLK_TDMA_GUARD_ASSERT_DISARMED(
        dfb_id, "TEN-4746: llk_push_tiles on a dfb with no pack (PACR) since llk_wait_for_free_tiles");
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    LLK_ASSERT(
        dfb_op_is_whole_share(local_dfb_interface, num_tiles),
        "llk_push_tiles: an op on a BLOCKED ring must move this hart's whole share");
    if (dfb_op_is_split<Pap, Cap, true>(local_dfb_interface)) {
        // Split: post each counter its share of the block and step every bookmark past it.
        const std::int32_t per_tc = num_tiles / local_dfb_interface.num_tcs_to_rr;
        for (std::uint8_t i = 0; i < local_dfb_interface.num_tcs_to_rr; i++) {
            TT_PUSH_TILES(PACK_SEL, per_tc, dfb::get_counter_id(local_dfb_interface.tc_slots[i].packed_tile_counter));
        }
        local_dfb_interface.wr_entry_ptr = 0;
        dfb_advance_all_slots<true>(local_dfb_interface, static_cast<std::uint32_t>(per_tc));
        return;
    }
    auto& slot = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx];
    std::uint32_t tc_id = dfb::get_counter_id(slot.packed_tile_counter);
    // Update the tile counters values
    TT_PUSH_TILES(PACK_SEL, num_tiles, tc_id);

    local_dfb_interface.wr_entry_ptr = 0;
    dfb_advance_slot<true>(local_dfb_interface, slot, num_tiles);
}

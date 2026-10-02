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
template <dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN, dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
inline void llk_wait_tiles(const std::int32_t dfb_id, const std::uint32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    LLK_ASSERT(
        dfb_op_is_whole_share(local_dfb_interface, num_tiles),
        "llk_wait_tiles: an op on a BLOCKED ring must move this hart's whole share");
    if (dfb_op_is_split<Pap, Cap, false>(local_dfb_interface)) {
        // Split: the block belongs to every counter, wait for each one's share.
        const std::uint32_t per_tc = num_tiles / local_dfb_interface.num_tcs_to_rr;
        for (std::uint8_t i = 0; i < local_dfb_interface.num_tcs_to_rr; i++) {
            TT_WAIT_TILES(
                ckernel::p_stall::STALL_UNPACK,
                per_tc,
                dfb::get_counter_id(local_dfb_interface.tc_slots[i].packed_tile_counter));
        }
    } else {
        std::uint32_t tc_id =
            dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
        TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, tc_id);
    }
    // TEN-4746: arm this dfb; a real unpack (UNPACR) on it must clear this before the matching pop.
    LLK_TDMA_GUARD_NOTE_WAIT(dfb_id);

    // TT_WAIT_TILES only gates the Tensix instruction stream and returns to the RISC-V core immediately. We want to
    // also block the RISC until that WAIT_TILES has resolved, so wait_front() has the same contract as on Blackhole:
    // when it returns, a RISC-side L1 read of the waited entries is safe. Poll this thread's SYNC busy bit in the
    // tensix_busy_status CSR.
    ckernel::wait_sync_idle();
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
template <
    std::uint8_t UNPACK_SEL = 0x3,
    dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN,
    dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
inline void llk_pop_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    // TEN-4746: popping a dfb that was waited but never unpacked (no UNPACR since wait_tiles) is a HW
    // hazard -- the wait can resolve before tiles are available.
    LLK_TDMA_GUARD_ASSERT_DISARMED(
        dfb_id, "TEN-4746: llk_pop_tiles on a dfb with no unpack (UNPACR) since llk_wait_tiles");
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    LLK_ASSERT(
        dfb_op_is_whole_share(local_dfb_interface, num_tiles),
        "llk_pop_tiles: an op on a BLOCKED ring must move this hart's whole share");
    if (dfb_op_is_split<Pap, Cap, false>(local_dfb_interface)) {
        // Split: ack each counter its share of the block and step every bookmark past it.
        const std::uint32_t per_tc = num_tiles / local_dfb_interface.num_tcs_to_rr;
        for (std::uint8_t i = 0; i < local_dfb_interface.num_tcs_to_rr; i++) {
            TT_POP_TILES(UNPACK_SEL, per_tc, dfb::get_counter_id(local_dfb_interface.tc_slots[i].packed_tile_counter));
        }
        dfb_advance_all_slots(local_dfb_interface, per_tc);
        return;
    }
    auto& slot = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx];
    std::uint32_t tc_id = dfb::get_counter_id(slot.packed_tile_counter);

    // Wait until selected unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, tc_id);

    dfb_advance_slot(local_dfb_interface, slot, num_tiles);
}

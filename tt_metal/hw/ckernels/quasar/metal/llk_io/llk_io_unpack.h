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
 * @brief  Wait for num_tiles available in the incoming dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
inline void llk_wait_tiles(const std::int32_t dfb_id, const std::uint32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    uint32_t tc_id = dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    // [TEN-4746 / #48552] Intervening TDMA op before the SYNC-class counter read. A bare
    // counter-mutation -> WAIT_TILES on the same tc_id races: the SYNC read reaches the tile-counter
    // engine before the prior TDMA counter op retires and latches a wrong-Neo counter address
    // (TILE_COUNTERS index 0x10000 fault, observed on the K-spill matmul via cb_intermed0, on UNPACK).
    // TEN-4746 requires an *intervening TDMA op* (not a STALLWAIT sync-stall, which does not fence the
    // counter engine); UNPACR_NOP is the ISA's documented "cycle delay between back-to-back unpack".
    TTI_UNPACR_NOP(ckernel::p_unpacr::UNP_A, 0, 0, 0, 0, ckernel::p_unpacr::UNP_NOP);
    TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, tc_id);
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
template <std::uint8_t UNPACK_SEL = 0x3>
inline void llk_pop_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(dfb_id);
    auto& slot = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx];
    uint32_t tc_id = dfb::get_counter_id(slot.packed_tile_counter);

    // Wait until selected unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, tc_id);
    // [TEN-4746 / #48552] Intervening TDMA op after the TDMA-class POP_TILES counter mutation so a
    // following counter op on the same tc_id cannot race its retire (see llk_wait_tiles above for the
    // full rationale). UNPACR_NOP is a real unpack-pipeline TDMA op; a STALLWAIT sync-stall does NOT
    // fence the counter engine (disproven: same 0x10000 fault persisted).
    TTI_UNPACR_NOP(ckernel::p_unpacr::UNP_A, 0, 0, 0, 0, ckernel::p_unpacr::UNP_NOP);

    dfb_advance_slot(local_dfb_interface, slot, num_tiles);
}

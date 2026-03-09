// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tools/profiler/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"

/**
 * @brief  Wait for num_tiles available in the incoming dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
inline void llk_wait_tiles(const std::int32_t dfb_id, const std::uint32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, tc_id);
    DPRINT << "llk_wait_tiles: dfb_id " << dfb_id << "tc_id: " << tc_id << ENDL();
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
template <std::uint8_t UNPACK_SEL = 0x3>
inline void llk_pop_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = dfb::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    // Wait until selected unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, tc_id);
    DPRINT << "llk_pop_tiles: dfb_id " << dfb_id << " tc_id: " << tc_id << ENDL();

    DPRINT << "acked " << static_cast<uint32_t>(ckernel::trisc::tile_counters[tc_id].f.acked) << " posted " << static_cast<uint32_t>(ckernel::trisc::tile_counters[tc_id].f.posted) << ENDL();

    // Update the DFB buffer information
    const std::uint32_t num_words = num_tiles * local_dfb_interface.stride_size;

    local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx +=
        local_dfb_interface.stride_size_tiles;
    local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr += num_words;
    if (local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr >=
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].limit) {
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr =
            local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].base_addr;
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx =
            local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].base_entry_idx;
    }

    local_dfb_interface.tc_idx = (local_dfb_interface.tc_idx + 1) % local_dfb_interface.num_tcs_to_rr;
}

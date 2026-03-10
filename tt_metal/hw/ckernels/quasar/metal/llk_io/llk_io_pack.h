// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tools/profiler/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/dataflow_buffer_init.h"
#include "api/debug/dprint.h"

static uint32_t g_pack_push_count = 0;

/**
 * @brief  Wait for num_tiles of free space in the dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles of free space to wait for in dataflow buffer
 */
inline void llk_wait_for_free_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    experimental::LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = experimental::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
    // uint8_t tensix_id =
    // experimental::get_tensix_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter); DPRINT
    // << "WAIT_FREE dfb=" << dfb_id << " tc=" << tc_id
    //        << " tensix_id=" << static_cast<uint32_t>(tensix_id)
    //        << " tc_idx=" << static_cast<uint32_t>(local_dfb_interface.tc_idx)
    //        << " wr_ptr=" << local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_ptr
    //        << " wr_entry_idx=" << local_dfb_interface.wr_entry_idx
    //        << " push_count=" << g_pack_push_count << ENDL();
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
    experimental::LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = experimental::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
    uint8_t tensix_id =
        experimental::get_tensix_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);
    // Update the tile counters values
    TT_PUSH_TILES(PACK_SEL, num_tiles, tc_id);
    g_pack_push_count++;
    DPRINT << "CMP PUSH: tc=" << static_cast<uint32_t>(tc_id) << " dfb=" << static_cast<uint32_t>(dfb_id)
           << " tensix_id=" << static_cast<uint32_t>(tensix_id) << " num_tiles=" << static_cast<uint32_t>(num_tiles)
           << " push_count=" << g_pack_push_count << " capacity=" << ckernel::trisc::tile_counters[tc_id].f.buf_capacity
           << " posted=" << ckernel::trisc::tile_counters[tc_id].f.posted
           << " acked=" << ckernel::trisc::tile_counters[tc_id].f.acked << ENDL();

    const std::uint32_t num_words = num_tiles * local_dfb_interface.stride_size;

    local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_ptr += num_words;
    local_dfb_interface.wr_entry_idx += num_tiles;
    local_dfb_interface.wr_entry_ptr = 0;

    bool wrapped =
        (local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_ptr ==
         local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].limit);
    if (wrapped) {
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_ptr = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].base_addr;
        if (local_dfb_interface.tc_idx == local_dfb_interface.num_tcs_to_rr - 1) {
            local_dfb_interface.wr_entry_idx = 0;
        }
    }

    local_dfb_interface.tc_idx = (local_dfb_interface.tc_idx + 1) % local_dfb_interface.num_tcs_to_rr;
    // DPRINT << "PACK push done: dfb=" << dfb_id
    //        << " push_count=" << g_pack_push_count
    //        << " wrapped=" << static_cast<uint32_t>(wrapped)
    //        << " new_wr_ptr=" << local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_ptr
    //        << " tc_idx=" << static_cast<uint32_t>(local_dfb_interface.tc_idx)
    //        << " num_tcs_to_rr=" << static_cast<uint32_t>(local_dfb_interface.num_tcs_to_rr) << ENDL();
}

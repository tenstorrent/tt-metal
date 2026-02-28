// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tools/profiler/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "internal/circular_buffer_interface.h"
#include "internal/dataflow_buffer_init.h"

/**
 * @brief  Wait for num_tiles available in the incoming dataflow buffer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
inline void llk_wait_tiles(const std::int32_t dfb_id, const std::uint32_t num_tiles) {
    experimental::LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = experimental::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, tc_id);
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param dfb_id: Dataflow Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in dataflow buffer
 */
template <std::uint8_t UNPACK_SEL = 0x3>
inline void llk_pop_tiles(const std::int32_t dfb_id, const std::int32_t num_tiles) {
    experimental::LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_id];
    uint32_t tc_id = experimental::get_counter_id(local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].packed_tile_counter);

    // Wait until selected unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, tc_id);

    // Update the DFB buffer information
    const std::uint32_t num_words = num_tiles * local_dfb_interface.stride_size;

    local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr += num_words;
    local_dfb_interface.rd_entry_idx += num_tiles;
    if (local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr == local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].limit) {
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_ptr = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].base_addr;
        // rd_entry_idx is a global index, not per-tc for a given DFB. Only reset it when we reached the limit for the entire buffer, not just for the current tc.
        if (local_dfb_interface.tc_idx == local_dfb_interface.num_tcs_to_rr - 1) {
            local_dfb_interface.rd_entry_idx = 0;
        }
    }

    local_dfb_interface.tc_idx = (local_dfb_interface.tc_idx + 1) % local_dfb_interface.num_tcs_to_rr;
}

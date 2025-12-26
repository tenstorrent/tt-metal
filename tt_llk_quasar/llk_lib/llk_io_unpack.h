// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "internal/circular_buffer_interface.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief  Wait for num_tiles available in the incoming circular buffer
 * @param cb_id: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in circular buffer
 */
inline void _llk_wait_tiles_(const std::int32_t cb_id, const uint32_t num_tiles)
{
    TT_WAIT_TILES(p_stall::STALL_UNPACK, num_tiles, cb_id);
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @param cb_id: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in circular buffer
 */
template <std::uint8_t UNPACK_SEL = 0x7>
inline void _llk_pop_tiles_(const std::int32_t cb_id, const std::int32_t num_tiles)
{
    // Wait until all 3 unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, cb_id);

    get_local_cb_interface(cb_id).tiles_acked += num_tiles;
    const std::uint32_t num_words = num_tiles * get_local_cb_interface(cb_id).fifo_page_size;
    get_local_cb_interface(cb_id).fifo_rd_ptr += num_words;

    if (get_local_cb_interface(cb_id).fifo_rd_ptr >= get_local_cb_interface(cb_id).fifo_limit)
    {
        get_local_cb_interface(cb_id).fifo_rd_ptr -= get_local_cb_interface(cb_id).fifo_size;
    }
}

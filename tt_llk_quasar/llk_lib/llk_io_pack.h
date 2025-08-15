// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "circular_buffer.h"
#include "ckernel.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief  Wait for num_tiles of free space in the circular buffer
 * @tparam CB_ID: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles of free space to wait for in circular buffer
 */
template <std::int32_t CB_ID>
inline void _llk_wait_for_free_tiles_(const std::int32_t num_tiles)
{
    static_assert((CB_ID < 32 && CB_ID >= 0), "CB_ID should be between 0-31");
    TT_WAIT_FREE(p_stall::STALL_PACK, num_tiles, CB_ID);
}

/**
 * @brief  Push num_tiles into the circular buffer, increment write pointer
 * @tparam CB_ID: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to push into circular buffer
 */
// Push N tiles to stream buffer (increment write pointer)
template <std::int32_t CB_ID, std::uint8_t PACK_SEL = 0x3>
inline void _llk_push_tiles_(const std::int32_t num_tiles)
{
    static_assert((CB_ID < 32 && CB_ID >= 0), "CB_ID should be between 0-31");

    // Update the tile counters values
    TT_PUSH_TILES(PACK_SEL, num_tiles, CB_ID);

    // Update the CB buffer information
    std::uint32_t num_words = num_tiles * get_local_cb_interface(CB_ID).fifo_page_size;

    get_local_cb_interface(CB_ID).tiles_received += num_tiles;
    get_local_cb_interface(CB_ID).fifo_wr_ptr += num_words;

    if (get_local_cb_interface(CB_ID).fifo_wr_ptr >= get_local_cb_interface(CB_ID).fifo_limit)
    {
        get_local_cb_interface(CB_ID).fifo_wr_ptr -= get_local_cb_interface(CB_ID).fifo_size;
    }
}

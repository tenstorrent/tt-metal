// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "circular_buffer.h"
#include "ckernel.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief  Wait for num_tiles available in the incoming circular buffer
 * @tparam CB_ID: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in circular buffer
 */
template <std::int32_t CB_ID>
inline void _llk_wait_tiles_(const uint32_t num_tiles)
{
    static_assert((CB_ID < 32 && CB_ID >= 0), "CB_ID should be between 0-31");
    TT_WAIT_TILES(p_stall::STALL_UNPACK, num_tiles, CB_ID);
}

/**
 * @brief Pop num_tiles tiles from the incoming stream, increment read pointer
 * @tparam CB_ID: Circular Buffer ID, values = [0-31]
 * @param num_tiles: Number of tiles to wait for in circular buffer
 */
template <std::int32_t CB_ID, std::uint8_t UNPACK_SEL = 0x7>
inline void _llk_pop_tiles_(const std::int32_t num_tiles)
{
    static_assert((CB_ID < 32 && CB_ID >= 0), "CB_ID should be between 0-31");

    // Wait until all 3 unpackers are reading from L1
    TT_POP_TILES(UNPACK_SEL, num_tiles, CB_ID);

    get_local_cb_interface(CB_ID).tiles_acked += num_tiles;
    const std::uint32_t num_words = num_tiles * get_local_cb_interface(CB_ID).fifo_page_size;
    get_local_cb_interface(CB_ID).fifo_rd_ptr += num_words;

    if (get_local_cb_interface(CB_ID).fifo_rd_ptr >= get_local_cb_interface(CB_ID).fifo_limit)
    {
        get_local_cb_interface(CB_ID).fifo_rd_ptr -= get_local_cb_interface(CB_ID).fifo_size;
    }
}

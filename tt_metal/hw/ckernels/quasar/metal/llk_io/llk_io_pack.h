// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "internal/circular_buffer_interface.h"
#include "llk_io_pack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    DeviceZoneScopedSumN2("CB-COMPUTE-RESERVE-BACK");
    // _llk_wait_for_free_tiles_(operand, num_tiles);
    TT_WAIT_FREE(p_stall::STALL_PACK, num_tiles, operand);
}

inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    // _llk_push_tiles_(operand, num_tiles);
    // Update the tile counters values
    DPRINT << "pushing tiles " << num_tiles << ENDL();
    TT_PUSH_TILES(0x3, num_tiles, operand);

    // Update the CB buffer information
    std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(operand).tiles_received += num_tiles;
    get_local_cb_interface(operand).fifo_wr_ptr += num_words;
    get_local_cb_interface(operand).fifo_wr_tile_ptr = 0;

    if (get_local_cb_interface(operand).fifo_wr_ptr >= get_local_cb_interface(operand).fifo_limit) {
        get_local_cb_interface(operand).fifo_wr_ptr -= get_local_cb_interface(operand).fifo_size;
    }
    get_local_cb_interface(operand).fifo_wr_tile_idx += num_tiles;
    if (get_local_cb_interface(operand).fifo_wr_tile_idx >= get_local_cb_interface(operand).fifo_num_pages) {
        get_local_cb_interface(operand).fifo_wr_tile_idx -= get_local_cb_interface(operand).fifo_num_pages;
    }
}

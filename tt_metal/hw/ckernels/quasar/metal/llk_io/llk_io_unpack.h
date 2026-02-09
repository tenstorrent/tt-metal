// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "internal/circular_buffer_interface.h"
#include "llk_io_unpack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    // _llk_wait_tiles_(operand, num_tiles);
    TT_WAIT_TILES(p_stall::STALL_UNPACK, num_tiles, operand);
}

inline void llk_pop_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    // _llk_pop_tiles_(operand, num_tiles);
    DPRINT << "popping tiles " << num_tiles << ENDL();
    TT_POP_TILES(0x7, num_tiles, operand);

    get_local_cb_interface(operand).tiles_acked += num_tiles;
    const std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;
    get_local_cb_interface(operand).fifo_rd_ptr += num_words;

    if (get_local_cb_interface(operand).fifo_rd_ptr >= get_local_cb_interface(operand).fifo_limit) {
        get_local_cb_interface(operand).fifo_rd_ptr -= get_local_cb_interface(operand).fifo_size;
    }
    get_local_cb_interface(operand).fifo_rd_tile_idx += num_tiles;
    if (get_local_cb_interface(operand).fifo_rd_tile_idx >= get_local_cb_interface(operand).fifo_num_pages) {
        get_local_cb_interface(operand).fifo_rd_tile_idx -= get_local_cb_interface(operand).fifo_num_pages;
    }
}

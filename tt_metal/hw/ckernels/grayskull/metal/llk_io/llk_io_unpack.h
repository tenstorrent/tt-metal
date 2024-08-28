// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "llk_unpack_common.h"
#include "tools/profiler/kernel_profiler.hpp"
using namespace ckernel;

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    // TODO(MO): Manually uncomment until issue #6619 is resolved
    //DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - cb_interface[input].tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {

    std::uint32_t input = operand;
    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words = num_tiles * cb_interface[operand].fifo_page_size;

    cb_interface[input].tiles_acked += num_tiles;
    TT_SETDMAREG(0, cb_interface[input].tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
    cb_interface[input].fifo_rd_ptr += num_words;

    if (cb_interface[input].fifo_rd_ptr >= cb_interface[input].fifo_limit) {
        cb_interface[input].fifo_rd_ptr -= cb_interface[input].fifo_size;
    }
}

inline void llk_wait_blocks(int operand, std::int32_t num_blocks) { llk_wait_tiles(operand, num_blocks); }

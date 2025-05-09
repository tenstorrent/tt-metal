// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "llk_unpack_common_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// This address corresponds to:
// auto buf = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
// uint32_t* data = buf->data;
// On tensix cores, on WH, this is equivalent to 0x001e4
// The size of the buffer is 32 4 byte values, and for trisc kernels we are using lower half of it.
// Number of valid elements to dump is 16 for trisc kernels
// MAKE SURE TO DOUBLE CHECK THIS VALUE WHEN REBASING
// FIXME MT: double check this
// volatile uint32_t* dbg_dump_trisc = (volatile uint32_t*)0x001e4;

using namespace ckernel;

// Apply delay on odd rows of tensix cores in case of matmul.
// We do this so that not all cores start at once, therefore reducing the chance of di/dt problems.
// MM_STAGGER_ODD_ROWS is an externally controlled define, set up in program compilation.
inline __attribute__((__always_inline__)) void apply_mm_stagger(int operand) {
#if MM_STAGGER_TYPE == 1  // first block stagger
    static bool stagger_applied = false;
    constexpr int stagger_operand = 1;
    constexpr int stagger_delay_in_cycles = MM_STAGGER_VALUE;
    if (stagger_applied == false && operand == stagger_operand) {
        stagger_applied = true;
        constexpr uint32_t noc_id = 0;
        uint32_t noc_id_logical_reg = NOC_CFG_READ_REG(noc_id, NOC_ID_LOGICAL);
        uint32_t my_logical_y = (noc_id_logical_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        // Apply stagger on odd rows only
        if (my_logical_y & 0x1) {
            wait(stagger_delay_in_cycles);
        }
    }
#elif MM_STAGGER_TYPE == 2  // every block stagger
    constexpr int stagger_operand = 1;
    constexpr int stagger_delay_in_cycles = MM_STAGGER_VALUE;
    if (operand == stagger_operand) {
        constexpr uint32_t noc_id = 0;
        uint32_t noc_id_logical_reg = NOC_CFG_READ_REG(noc_id, NOC_ID_LOGICAL);
        uint32_t my_logical_y = (noc_id_logical_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        // Apply stagger on odd rows only
        if (my_logical_y & 0x1) {
            wait(stagger_delay_in_cycles);
        }
    }
#elif MM_STAGGER_TYPE == 3  // hybrid stagger - apply stagger every 20th block
    constexpr int stagger_operand = 1;
    constexpr int stagger_delay_in_cycles = MM_STAGGER_VALUE;
    static int stagger_counter = 0;
    if (operand == stagger_operand) {
        if ((stagger_counter % 20) == 0) {
            constexpr uint32_t noc_id = 0;
            uint32_t noc_id_logical_reg = NOC_CFG_READ_REG(noc_id, NOC_ID_LOGICAL);
            uint32_t my_logical_y = (noc_id_logical_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
            // Apply stagger on odd rows only
            if (my_logical_y & 0x1) {
                wait(stagger_delay_in_cycles);
            }
        }
        stagger_counter++;
    }
#endif
}

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    // TODO(MO): Manually uncomment until issue #6619 is resolved
    // DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    if (operand == 0) {
        // FIXME MT:*(dbg_dump_trisc + 3) = 0x00baba01;
    } else {
        // FIXME MT:*(dbg_dump_trisc + 4) = 0x00baba01;
    }
    // FIXME MT:*(dbg_dump_trisc + 5) = (uint32_t)get_local_cb_interface(input).fifo_rd_ptr;
    // FIXME MT:*(dbg_dump_trisc + 6) = (uint32_t)get_local_cb_interface(input).fifo_size;
    // FIXME MT:*(dbg_dump_trisc + 7) = (uint32_t)get_local_cb_interface(input).fifo_limit;
    // FIXME MT:*(dbg_dump_trisc + 8) = (uint32_t)get_local_cb_interface(input).tiles_acked;
    // FIXME MT:*(dbg_dump_trisc + 9) = (uint32_t)get_local_cb_interface(input).fifo_page_size;
    // FIXME MT:*(dbg_dump_trisc + 10) = (uint32_t)tiles_received_ptr;

    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - get_local_cb_interface(input).tiles_acked;
    } while (num_tiles_recv < num_tiles_u);

    if (operand == 0) {
        // FIXME MT:*(dbg_dump_trisc + 3) = 0x08baba01;
    } else {
        // FIXME MT:*(dbg_dump_trisc + 4) = 0x08baba01;
    }

    apply_mm_stagger(operand);

    // FIXME MT:if (operand == 0) {
    // FIXME MT:    uint block_id = *(dbg_dump_trisc + 1);
    // FIXME MT:    if (block_id == 0) {
    // FIXME MT:        uint iter_id = *(dbg_dump_trisc + 2);
    // FIXME MT:        *(dbg_dump_trisc + 2) = iter_id + 1;
    // FIXME MT:    }
    // FIXME MT:    *(dbg_dump_trisc + 1) = block_id + 1;
    // FIXME MT:}
}

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {
    std::uint32_t input = operand;
    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(input).tiles_acked += num_tiles;
    TT_SETDMAREG(0, get_local_cb_interface(input).tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
    get_local_cb_interface(input).fifo_rd_ptr += num_words;

    if (get_local_cb_interface(input).fifo_rd_ptr >= get_local_cb_interface(input).fifo_limit) {
        get_local_cb_interface(input).fifo_rd_ptr -= get_local_cb_interface(input).fifo_size;
    }
}

inline void llk_wait_blocks(int operand, std::int32_t num_blocks) { llk_wait_tiles(operand, num_blocks); }

// FIXME-WH-UPLIFT
// FIXME: FP32 accumulation --> pop tiles in the operand? just change rd_ptr?
inline void llk_clear_tiles(std::uint32_t operand, std::uint32_t num_tiles) {
    // std::uint32_t input = operand_to_input_index(operand);
    // if (get_local_cb_interface(input).accumulation_buffer) {
    //     std::uint32_t num_words = num_tiles * get_local_cb_interface(input).fifo_page_size;

    //     get_local_cb_interface(input).fifo_rd_ptr += num_words;

    //     if (get_local_cb_interface(input).f.fifo_rd_ptr >= operands[input].fifo_limit) {
    //         get_local_cb_interface(input).f.fifo_rd_ptr -= operands[input].fifo_size;
    //     }

    //     get_local_cb_interface(input).f.fifo_rd_base_ptr = operands[input].fifo_rd_ptr; //inc base ptr

    //     get_local_cb_interface(input).curr_iter = 0;
    // }
}

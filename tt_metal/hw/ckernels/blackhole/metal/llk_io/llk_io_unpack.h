// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "llk_unpack_common_api.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "llk_assert.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

// Apply delay on odd rows of tensix cores in case of matmul.
// We do this so that not all cores start at once, therefore reducing the chance of di/dt problems.
// The hook runs after each wait on the in1 operand; what one wait covers is up to the kernel: the block-based
// matmul kernels wait on a whole in1 block, bmm_metal2.cpp waits on in1 one tile at a time, so there the delay
// is per tile.
// Controlled by three defines the host adds to the matmul compute kernel (compute_throttle_utils.cpp), the first
// two from the TT_MM_STAGGER_TYPE / TT_MM_STAGGER_VALUE environment variables:
//   MM_STAGGER_TYPE    1 = delay after the first wait on in1 only, 2 = after every wait,
//                      3 = after the first wait and every stagger_period_in_blocks-th after it
//                      (the wait counter is checked before it is incremented).
//   MM_STAGGER_VALUE   = delay in cycles.
//   MM_STAGGER_OPERAND = the kernel's operand id for in1: a CB index, or for a Metal 2.0 kernel the slot
//                        constant kernel_bindings_generated.h defines for the in1 DFB binding (DFB_SLOT_in1),
//                        since a DFB's device slot follows declaration order and is not always 1. The dfb::
//                        token cannot be used: that header declares it only after pulling this one in.
// Without the defines this function is empty; any other MM_STAGGER_TYPE, or a type without an operand, is a
// compile error.
#if defined(MM_STAGGER_TYPE) && !defined(MM_STAGGER_OPERAND)
#error \
    "MM_STAGGER_OPERAND (the compute kernel's in1 CB index, or DFB_SLOT_<name> for a Metal 2.0 kernel) must be defined with MM_STAGGER_TYPE"
#endif
inline __attribute__((__always_inline__)) void apply_mm_stagger(int operand) {
#if MM_STAGGER_TYPE == 1  // first block stagger
    static bool stagger_applied = false;
    constexpr int stagger_operand = MM_STAGGER_OPERAND;
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
    constexpr int stagger_operand = MM_STAGGER_OPERAND;
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
#elif MM_STAGGER_TYPE == 3  // hybrid stagger - delay the first wait on in1 and every stagger_period_in_blocks-th
                            // after it (the counter is checked before the increment)
    constexpr int stagger_operand = MM_STAGGER_OPERAND;
    constexpr int stagger_delay_in_cycles = MM_STAGGER_VALUE;
    constexpr int stagger_period_in_blocks = 20;
    static int stagger_counter = 0;
    if (operand == stagger_operand) {
        if ((stagger_counter % stagger_period_in_blocks) == 0) {
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
#elif defined(MM_STAGGER_TYPE)
#error "MM_STAGGER_TYPE must be 1, 2 or 3"
#endif
}

// Wait for N tiles available in the incoming stream.
// With the MM_STAGGER_* defines set it also spins the stagger delay after a wait on the in1 operand
// (MM_STAGGER_OPERAND) for odd-row cores (see apply_mm_stagger).
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;

    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    {
        SYNC_WAIT("SYNC-CB-WAIT", operand);
        do {
            tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
            num_tiles_recv = tiles_received - get_local_cb_interface(input).tiles_acked;
        } while (num_tiles_recv < num_tiles_u);
    }

    apply_mm_stagger(operand);
}

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {
    SYNC_SIGNAL("SYNC-CB-POP", operand);
    std::uint32_t input = operand;

    // Convert the counter's byte address to a Tensix 4-byte word address, masked to 18 bits.
    const std::uint32_t tiles_acked_addr_tensix =
        static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(input).tiles_acked += num_tiles;
    TT_SETDMAREG(0, get_local_cb_interface(input).tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, tiles_acked_addr_tensix);
    auto& cb = get_local_cb_interface(input);

    LLK_ASSERT(cb.fifo_rd_ptr < cb.fifo_limit, "CB pop_front: fifo_rd_ptr already at or past fifo_limit");

    std::uint32_t remaining = cb.fifo_limit - cb.fifo_rd_ptr;

    LLK_ASSERT(remaining >= num_words, "CB pop_front: fifo_rd_ptr would exceed fifo_limit");

    cb.fifo_rd_ptr += num_words;

    if (cb.fifo_rd_ptr >= cb.fifo_limit) {
        cb.fifo_rd_ptr -= cb.fifo_size;
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

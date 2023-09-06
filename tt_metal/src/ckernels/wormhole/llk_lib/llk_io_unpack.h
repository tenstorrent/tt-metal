/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "stream_interface.h"
#ifdef PERF_DUMP
#include "ckernel_perf_unpack_pack.h"
#endif

using namespace ckernel;

inline void llk_setup_operands() {
    while (l1_read_barrier(&EPOCH_INFO_PTR->all_streams_ready) == 0)
        ;
    for (std::uint32_t n = 0; n < l1_read_barrier(&EPOCH_INFO_PTR->num_inputs); n++) {
        // iterates through all streams that have stream->unpacker functionality (inputs + intermediates)
        std::uint32_t stream_id = l1_read_barrier(&EPOCH_INFO_PTR->inputs[n]->stream_id);
        std::uint32_t fifo_addr = stream_get_data_buf_addr(stream_id);
        std::uint32_t fifo_size = stream_get_data_buf_size(stream_id);
        mem_barrier(fifo_size);
        std::uint32_t input = operand_to_input_index(stream_id_to_operand(stream_id));
        cb_read_interface[input].fifo_rd_ptr = fifo_addr;
        cb_read_interface[input].fifo_rd_base_ptr = fifo_addr; // used for reads from interm buffers only
        cb_read_interface[input].fifo_size = fifo_size;
        cb_read_interface[input].fifo_limit = fifo_addr + fifo_size - 1;  // Check if there is overflow
        cb_read_interface[input].tiles_acked = 0;
        cb_read_interface[input].words_acked = 0;
        cb_read_interface[input].blocks_per_iter = 0; // number of ublocks written per k into interm buffer
        cb_read_interface[input].curr_block = 0; // current number of ublocks written into interm buffer
        cb_read_interface[input].num_iter = 0; // number of passes through interm buffer (aka k-1)
        cb_read_interface[input].curr_iter = 0; // current number of passes through interm buffer
        std::uint32_t operand = stream_id_to_operand(stream_id);
        if (operand_is_intermediate(operand)) {
           cb_read_interface[input].blocks_per_iter = l1_read_barrier(&EPOCH_INFO_PTR->mblock_m) * l1_read_barrier(&EPOCH_INFO_PTR->mblock_n);
           cb_read_interface[input].num_iter = l1_read_barrier(&EPOCH_INFO_PTR->mblock_k)-1;
        }
    }
}

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    std::uint32_t input = operand_to_input_index(operand);
    volatile std::uint32_t* tiles_received_ptr = get_operand_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;
    std::uint16_t tiles_received;
#if defined(PERF_DUMP) && PERF_DUMP_LEVEL > 0
    tiles_received = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_received_ptr);
    uint16_t num_tiles_recv = tiles_received - cb_read_interface[input].tiles_acked;
    if (num_tiles_recv < num_tiles_u) {
        if (record_perf_events) {
            uint32_t event_id = perf::get_event_id(
                operand, num_tiles, perf::EventType::WAIT_FOR_INCOMING_TILES, current_outer_loop_iter);
            record_timestamp_64b(event_id, 6); // Leave space for the end of last unpack wait-for-tile, its upper 32b, and num-tiles.
        }
       do {
            tiles_received = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_received_ptr);
            uint16_t num_tiles_recv = tiles_received - cb_read_interface[input].tiles_acked;
	    if (num_tiles_recv >= num_tiles_u) break;
        } while (1);
        if (record_perf_events) {
            uint32_t event_id = perf::get_event_id(
                operand, num_tiles, perf::EventType::WAIT_FOR_INCOMING_TILES, current_outer_loop_iter);
            record_timestamp_64b(event_id, 6); // Leave space for the end of last unpack wait-for-tile, its upper 32b, and num-tiles.
        }
    }
    mem_barrier(tiles_received);
#else

    do {
        tiles_received = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_received_ptr);
        uint16_t num_tiles_recv = tiles_received - cb_read_interface[input].tiles_acked;
	if (num_tiles_recv >= num_tiles_u) break;
    } while (1);
#endif

    volatile std::uint32_t* phase_changed_ptr = get_operand_phase_changed_ptr(operand);
    if (reg_read_barrier((std::uint32_t)phase_changed_ptr)) {
        cb_read_interface[input].f.fifo_rd_base_ptr = operands[input].f.fifo_limit - operands[input].fifo_size + 1;
        cb_read_interface[input].f.fifo_rd_ptr = operands[input].fifo_rd_base_ptr;
        *phase_changed_ptr = 0;
    }
}

// Pop N tiles from the incoming stream
template <bool pop_blocks = false>
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {
    std::uint32_t input = operand_to_input_index(operand);
    volatile std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_operand_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words;

    if constexpr (pop_blocks) {
        num_words = num_tiles *
                    ((32 * SCALE_DATUM_SIZE((uint)unpack_src_format[input], block_c_dim)) / 16);  // 32 rows per block
    } else {
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[input]);
    }

    cb_read_interface[input].tiles_acked += num_tiles;
    TT_SETDMAREG(0, cb_read_interface[input].tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);

    cb_read_interface[input].fifo_rd_ptr += num_words;

    if (cb_read_interface[input].f.fifo_rd_ptr > operands[input].fifo_limit) {
        cb_read_interface[input].f.fifo_rd_ptr -= operands[input].fifo_size;
    }

    if (operand_is_intermediate(operand)) {
       cb_read_interface[input].curr_block++;
       if (cb_read_interface[input].f.curr_block == operands[input].blocks_per_iter) {
          cb_read_interface[input].curr_iter++;
          if (cb_read_interface[input].f.curr_iter == operands[input].num_iter) {
             cb_read_interface[input].curr_iter=0;
             cb_read_interface[input].f.fifo_rd_base_ptr = operands[input].fifo_rd_ptr; //inc base ptr
          } else {
             cb_read_interface[input].f.fifo_rd_ptr = operands[input].fifo_rd_base_ptr; //set rd prt to base ptr
          }
          cb_read_interface[input].curr_block=0;
       }
    }
}

inline void llk_wait_blocks(int operand, std::int32_t num_blocks) { llk_wait_tiles(operand, num_blocks); }

// Pop N blocks from the incoming stream
inline void llk_pop_blocks(const std::int32_t operand, std::int32_t num_blocks, const std::int32_t block_c_dim) {
    llk_pop_tiles<true>(operand, num_blocks, block_c_dim);
}

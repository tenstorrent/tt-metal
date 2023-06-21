#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "epoch.h"
#include "stream_interface.h"
#include "llk_unpack_common.h"
#ifdef PERF_DUMP
#include "ckernel_perf_unpack_pack.h"
#endif

using namespace ckernel;

inline void llk_setup_operands() {
    while (EPOCH_INFO_PTR->all_streams_ready == 0)
        ;
    for (std::uint32_t n = 0; n < EPOCH_INFO_PTR->num_inputs; n++) {
        // iterates through all streams that have stream->unpacker functionality (inputs + intermediates)
        std::uint32_t stream_id = EPOCH_INFO_PTR->inputs[n]->stream_id;
        std::uint32_t fifo_addr = stream_get_data_buf_addr(stream_id);
        std::uint32_t fifo_size = stream_get_data_buf_size(stream_id);
        std::uint32_t input = operand_to_input_index(stream_id_to_operand(stream_id));
        operands[input].f.fifo_rd_ptr = fifo_addr;
        operands[input].f.fifo_rd_base_ptr = fifo_addr; // used for reads from interm buffers only
        operands[input].f.fifo_size = fifo_size;
        operands[input].f.fifo_limit = fifo_addr + fifo_size - 1;  // Check if there is overflow
        operands[input].f.tiles_acked = 0;
        operands[input].f.words_acked = 0;
        operands[input].f.blocks_per_iter = 0; // number of ublocks written per k into interm buffer
        operands[input].f.curr_block = 0; // current number of ublocks written into interm buffer
        operands[input].f.num_iter = 0; // number of passes through interm buffer (aka k-1)
        operands[input].f.curr_iter = 0; // current number of passes through interm buffer

        volatile std::uint32_t tt_reg_ptr * phase_changed_ptr = get_operand_phase_changed_ptr(stream_id_to_operand(stream_id));
        *phase_changed_ptr = 0;

        std::uint32_t operand = stream_id_to_operand(stream_id);
        if (operand_is_intermediate(operand)) {
           operands[input].f.blocks_per_iter = EPOCH_INFO_PTR->mblock_m * EPOCH_INFO_PTR->mblock_n;
           operands[input].f.num_iter = EPOCH_INFO_PTR->mblock_k-1;
           std::uint8_t packer_operand = EPOCH_INFO_PTR->inputs[n]->packer_operand;
           bool shared_buffer =  (packer_operand >= OPERAND_OUTPUT_START_INDEX);
           operands[input].f.accumulation_buffer = shared_buffer || // buffer sharing is enabled only if there is interm buffer with mblock_k>1
                                                   (!shared_buffer && (EPOCH_INFO_PTR->num_outputs==1)); // gradient accumulation buffer
                                              
        }
    }
}

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    
    std::uint32_t input = operand_to_input_index(operand);
    volatile std::uint32_t tt_reg_ptr * tiles_received_ptr = get_operand_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;
    std::uint16_t tiles_received;

#if defined(PERF_DUMP)
#if OVERLAY_INPUT_DECOUPLE == 1
    bool wait_for_tile_en = !is_input_operand_decoupled(operand); //operand_is_intermediate(operand);
#else
    bool wait_for_tile_en = !(DECOUPLINGS_EN && operand_is_intermediate(operand));
#endif
    if (wait_for_tile_en) {
    #if PERF_DUMP_LEVEL > 0
        tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
        uint16_t num_tiles_recv = tiles_received - operands[input].f.tiles_acked;
        if (num_tiles_recv < num_tiles_u) {
            uint32_t event_id = perf::get_event_id(
                operand, num_tiles, perf::EventType::WAIT_FOR_INCOMING_TILES, current_outer_loop_iter);
            record_timestamp_64b(event_id, 3); // Leave space for first-unpack-instruction
            do {
                tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
                uint16_t num_tiles_recv = tiles_received - operands[input].f.tiles_acked;
            if (num_tiles_recv >= num_tiles_u) break;
            } while (1);
            record_timestamp_64b(event_id, 3); // Leave space for first-unpack-instruction
        }
    #else
        do {
            tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
            uint16_t num_tiles_recv = tiles_received - operands[input].f.tiles_acked;
        if (num_tiles_recv >= num_tiles_u) break;
        } while (1);
    #endif
    }
#else

    do {
        tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
        uint16_t num_tiles_recv = tiles_received - operands[input].f.tiles_acked;
	if (num_tiles_recv >= num_tiles_u) break;
    } while (1);
#endif

    volatile std::uint32_t tt_reg_ptr * phase_changed_ptr = get_operand_phase_changed_ptr(operand);
    if (reg_read((std::uint32_t)phase_changed_ptr)) {
        operands[input].f.fifo_rd_base_ptr = operands[input].f.fifo_limit - operands[input].f.fifo_size + 1;
        operands[input].f.fifo_rd_ptr = operands[input].f.fifo_rd_base_ptr;
        *phase_changed_ptr = 0;
    }
#if defined(PERF_DUMP)
    record_latest_wait_for_tile();
#endif

}

inline void update_tiles_acked_ptr(const std::int32_t operand, const std::int32_t num_tiles, std::uint32_t input) {
    volatile std::uint32_t tt_reg_ptr * tiles_acked_ptr =
        (volatile std::uint32_t tt_reg_ptr *)((((volatile std::uint32_t)get_operand_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    operands[input].f.tiles_acked += num_tiles;
    TT_SETDMAREG(0, operands[input].f.tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
}

// Pop N tiles from the incoming stream
template <bool pop_blocks = false>
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles) {

    std::uint32_t input = operand_to_input_index(operand);
    std::uint32_t num_words;

    if constexpr (pop_blocks) {
        num_words = num_tiles * GET_L1_HEADERLESS_TILE_SIZE((uint)unpack_src_format[input]);
    } else {
        num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[input]);
    }

#if defined(PERF_DUMP)
    #if SKIP_UNP == 1 
        if (!operand_is_intermediate(operand)) {
            volatile std::uint32_t tt_reg_ptr * tiles_acked_ptr = get_operand_tiles_acked_ptr(operand);
            operands[input].f.tiles_acked += num_tiles;
            tiles_acked_ptr[0] = operands[input].f.tiles_acked & 0xffff;
        }
    #elif OVERLAY_INPUT_DECOUPLE == 1
        if (!is_input_operand_decoupled(operand)) {
            update_tiles_acked_ptr(operand, num_tiles, input);
        }
    #else
        update_tiles_acked_ptr(operand, num_tiles, input);
    #endif
#else
    update_tiles_acked_ptr(operand, num_tiles, input);
#endif

    operands[input].f.fifo_rd_ptr += num_words;

    if (operands[input].f.fifo_rd_ptr > operands[input].f.fifo_limit) {
        operands[input].f.fifo_rd_ptr -= operands[input].f.fifo_size;
    }

    if (operand_is_intermediate(operand)) {
        if (operands[input].f.accumulation_buffer) {
            operands[input].f.curr_block++;
            if (operands[input].f.curr_block == operands[input].f.blocks_per_iter) {
               operands[input].f.curr_iter++;
               if (operands[input].f.curr_iter == operands[input].f.num_iter) {
                  operands[input].f.curr_iter=0;
                  operands[input].f.fifo_rd_base_ptr = operands[input].f.fifo_rd_ptr; //inc base ptr
               } else {
                  operands[input].f.fifo_rd_ptr = operands[input].f.fifo_rd_base_ptr; //set rd prt to base ptr
               } 
               operands[input].f.curr_block=0;
            }
        }    
    }
}

inline void llk_clear_tiles(std::uint32_t operand, std::uint32_t num_tiles) {
    std::uint32_t input = operand_to_input_index(operand);
    if (operands[input].f.accumulation_buffer) {
        std::uint32_t num_words = num_tiles * GET_L1_TILE_SIZE((uint)unpack_src_format[input]);

        operands[input].f.fifo_rd_ptr += num_words;

        if (operands[input].f.fifo_rd_ptr > operands[input].f.fifo_limit) {
            operands[input].f.fifo_rd_ptr -= operands[input].f.fifo_size;
        }

        operands[input].f.fifo_rd_base_ptr = operands[input].f.fifo_rd_ptr; //inc base ptr

        operands[input].f.curr_iter = 0;
    }    
}  

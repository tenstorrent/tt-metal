#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "epoch.h"
#include "ckernel_gpr_map.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "llk_pack_common.h"
#ifdef PERF_DUMP
#include "ckernel_perf_unpack_pack.h"
#endif

using namespace ckernel;

inline void llk_setup_outputs() {
    while (EPOCH_INFO_PTR->all_streams_ready == 0)
        ;
    for (std::uint32_t n = 0; n < EPOCH_INFO_PTR->num_outputs; n++) {
        std::uint32_t stream_id = EPOCH_INFO_PTR->outputs[n]->stream_id;
        std::uint32_t fifo_addr = EPOCH_INFO_PTR->outputs[n]->buf_base_addr/MEM_WORD_BYTES;
        std::uint32_t fifo_size = EPOCH_INFO_PTR->outputs[n]->buf_full_size_bytes/MEM_WORD_BYTES;
        std::uint32_t fifo_size_tiles = EPOCH_INFO_PTR->outputs[n]->buf_size_tiles;
        std::int32_t operand = stream_id_to_operand(stream_id);
        bool is_intermediate_operand = operand >= OPERAND_INTERMEDIATES_START_INDEX && operand < OPERAND_RELAY_START_INDEX;
        volatile std::uint32_t tt_reg_ptr * tiles_received_ptr = get_operand_tiles_received_ptr(operand);
        std::uint32_t output = operand_to_output_index(operand);
        std::uint8_t legacy_pack = EPOCH_INFO_PTR->outputs[n]->legacy_pack;
        std::uint32_t dram_output_no_push = (EPOCH_INFO_PTR->outputs[n]->flags & STREAM_DRAM_NO_PUSH) != 0 ? 1 : 0;
        outputs[output].f.fifo_wr_ptr = fifo_addr;
        outputs[output].f.fifo_limit = fifo_addr + fifo_size - 1;  // Check if there is overflow
        outputs[output].f.fifo_size = fifo_size;
        outputs[output].f.fifo_wr_tile_ptr = 0;
        outputs[output].f.fifo_size_tiles = fifo_size_tiles;
        outputs[output].f.tiles_received = is_intermediate_operand ? reg_read((uint32_t)&tiles_received_ptr[0]) : 0;
        outputs[output].f.legacy_pack = legacy_pack;
        outputs[output].f.fifo_wr_base_ptr = fifo_addr;
        outputs[output].f.tile_size_words = (std::uint16_t)EPOCH_INFO_PTR->outputs[n]->tile_size_words;

        stream_should_packer_reset_pointers(stream_id);

        // untilize into interm buffer is not supported
        if (is_intermediate_operand) {
           std::uint8_t packer_operand = EPOCH_INFO_PTR->outputs[n]->packer_operand;
           outputs[output].f.shared_buffer =  packer_operand >= OPERAND_OUTPUT_START_INDEX;
           outputs[output].f.shared_buffer_operand = packer_operand;

           outputs[output].f.accumulation_buffer = outputs[output].f.shared_buffer || // buffer sharing is enabled only if there is interm buffer with mblock_k>1
                                                   (!outputs[output].f.shared_buffer && (EPOCH_INFO_PTR->num_outputs==1)); // gradient accumulation buffer

           outputs[output].f.blocks_per_iter = EPOCH_INFO_PTR->mblock_m * EPOCH_INFO_PTR->mblock_n;
           outputs[output].f.num_iter = EPOCH_INFO_PTR->mblock_k-1;
           outputs[output].f.curr_block = 0;
           outputs[output].f.curr_iter = 0;
           outputs[output].f.out_tile_dim =  outputs[output].f.blocks_per_iter * EPOCH_INFO_PTR->ublock_ct * EPOCH_INFO_PTR->ublock_rt;
        } else {
           outputs[output].f.shared_buffer = 0;
           outputs[output].f.accumulation_buffer = 0;

           outputs[output].f.ublock_ct = EPOCH_INFO_PTR->ublock_ct;
           outputs[output].f.ublock_tile_dim = EPOCH_INFO_PTR->ublock_ct * EPOCH_INFO_PTR->ublock_rt;
           outputs[output].f.row_tile_dim = EPOCH_INFO_PTR->ublock_ct * EPOCH_INFO_PTR->mblock_n;
           outputs[output].f.block_tile_dim = outputs[output].f.ublock_tile_dim * EPOCH_INFO_PTR->mblock_n;
           outputs[output].f.ublock_tile_cnt = 0;
           outputs[output].f.block_tile_cnt = 0;
        }
        // Get output fork info
        std::uint16_t num_fork_streams = EPOCH_INFO_PTR->outputs[n]->num_fork_streams;
        if (num_fork_streams > 0 && legacy_pack) {
            outputs[output].f.fork = 1;
            outputs[output].f.num_fork_streams = num_fork_streams;
            for (std::uint32_t k = 0; k < num_fork_streams; k++) {
                volatile epoch_stream_info_t tt_l1_ptr * fork_stream_info = EPOCH_INFO_PTR->active_streams[EPOCH_INFO_PTR->outputs[n]->fork_idxs[k]];
                std::uint8_t fork_stream_id = fork_stream_info->stream_id;
                outputs[output].f.fork_stream_ids[k] = fork_stream_id;
                dram_output_no_push |= (fork_stream_info->flags & STREAM_DRAM_NO_PUSH) != 0 ? (1 << (1+k)) : 0;
            }
        } else {
            outputs[output].f.fork = 0;
        }
        outputs[output].f.dram_output_no_push = dram_output_no_push;
    }
}

void stream_wait_for_free_tiles(std::uint32_t operand, std::uint32_t stream_id, std::int32_t num_tiles, std::uint32_t num_words, std::uint32_t fork_indx, std::uint32_t dram_output_no_push) {
  bool dram_output_no_push_bool = (dram_output_no_push & (1 << fork_indx)) ? 1 : 0;

  if (dram_output_no_push_bool) {
    volatile std::uint32_t tt_reg_ptr * tiles_acked_ptr = get_operand_tiles_acked_ptr(stream_id_to_operand(stream_id));
    std::uint32_t output = operand_to_output_index(operand);
    std::int32_t free_tiles;
#if defined(PERF_DUMP) && PERF_DUMP_LEVEL > 0
    std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
    std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
    free_tiles = (std::int32_t) free_tiles_wrap;
    if (free_tiles < num_tiles) {
        uint32_t event_id = perf::get_event_id(
            operand, num_tiles, perf::EventType::WAIT_FOR_FREE_TILES, current_outer_loop_iter);
        record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
        do {
            tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
            free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
            free_tiles = (std::int32_t) free_tiles_wrap;
        } while (free_tiles < num_tiles);
        record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
    }
#else
    do {
        std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
        std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
        free_tiles = (std::int32_t) free_tiles_wrap;
    } while (free_tiles < num_tiles);
#endif
  }

  std::uint32_t free_words;
  bool stream_not_ready = true;
#if defined(PERF_DUMP) && PERF_DUMP_LEVEL > 0
  free_words = stream_get_free_words(stream_id);
  if (dram_output_no_push_bool)
    stream_not_ready = free_words == 0;
  else
    stream_not_ready = free_words < num_words;
  if (stream_not_ready) {
      uint32_t event_id = perf::get_event_id(
          operand, num_tiles, perf::EventType::WAIT_FOR_FREE_TILES, current_outer_loop_iter);
      record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
      do {
          free_words = stream_get_free_words(stream_id);
          if (dram_output_no_push_bool)
              stream_not_ready = free_words == 0;
          else
              stream_not_ready = free_words < num_words;
      } while (stream_not_ready);
      record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
  }
#else
  do {
      free_words = stream_get_free_words(stream_id);
      if (dram_output_no_push_bool)
        stream_not_ready = free_words == 0;
      else
        stream_not_ready = free_words < num_words;
  } while (stream_not_ready);
#endif
}

// Blocking call to wait for free space needed to pack N tiles
template <bool skip_sync = false, bool wait_for_blocks = false, bool brisc_pack = false>
inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {

    std::uint32_t output = operand_to_output_index(operand);
    std::uint32_t num_words = num_tiles * (std::uint32_t)outputs[output].f.tile_size_words;
    bool legacy_pack = false;
    std::uint32_t dram_output_no_push = outputs[output].f.dram_output_no_push;
    if constexpr (brisc_pack) {
       legacy_pack = outputs[output].f.legacy_pack;
    } 

    if constexpr (wait_for_blocks) {
        num_words = num_tiles * (std::uint32_t)outputs[output].f.tile_size_words;
    }

    volatile std::uint32_t tt_reg_ptr * tiles_acked_ptr;
    if (brisc_pack && !legacy_pack) {
        tiles_acked_ptr = get_packer_tiles_acked_ptr(operand);
    } else {
        tiles_acked_ptr = get_operand_tiles_acked_ptr(operand);
    }

#if defined(PERF_DUMP)
    bool wait_for_tile_en = true;
#if OVERLAY_OUTPUT_DECOUPLE == 1
    wait_for_tile_en = !is_output_operand_decoupled(operand);
#endif
#endif

    if (operand_is_intermediate(operand)) {
        // Interm buffer always has space when we do inline packing
        //std::int32_t free_tiles;
        //do {
        //    std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
        //    std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (tiles_acked - outputs[output].f.tiles_received);
        //    free_tiles = (std::int32_t) free_tiles_wrap;
        //} while (free_tiles < num_tiles);

        // Make sure there is space in shared output buffer to fit entire block
        if (outputs[output].f.shared_buffer) {
           if ((0 == outputs[output].f.curr_iter) &&
               (0 == outputs[output].f.curr_block)) {
              std::uint32_t shared_output = operand_to_output_index(outputs[output].f.shared_buffer_operand);
              std::uint32_t prev_fifo_wr_ptr = outputs[shared_output].f.fifo_wr_ptr;
#if defined(PERF_DUMP) && DECOUPLINGS_EN
#else
              llk_wait_for_free_tiles<true, wait_for_blocks, brisc_pack>(outputs[output].f.shared_buffer_operand, outputs[output].f.out_tile_dim);
#endif
              if (outputs[shared_output].f.fifo_wr_ptr != prev_fifo_wr_ptr) {
                // Force unpacker to reset phase when packer resets phase
                volatile std::uint32_t tt_reg_ptr * phase_changed_ptr = get_operand_phase_changed_ptr(operand);
                *phase_changed_ptr = 1;
                outputs[output].f.fifo_wr_base_ptr = outputs[output].f.fifo_limit - outputs[output].f.fifo_size + 1;
                outputs[output].f.fifo_wr_ptr = outputs[output].f.fifo_wr_base_ptr;
              }
           } 
        }    
    } else if (wait_for_blocks || (brisc_pack && !legacy_pack)) {
        std::int32_t free_tiles;
#if defined(PERF_DUMP)
        if (wait_for_tile_en) {
    #if PERF_DUMP_LEVEL > 0
            std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
            std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
            free_tiles = (std::int32_t) free_tiles_wrap;
            if (free_tiles < num_tiles) {
                uint32_t event_id = perf::get_event_id(
                    operand, num_tiles, perf::EventType::WAIT_FOR_FREE_TILES, current_outer_loop_iter);
                record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
                do {
                    tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
                    free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
                    free_tiles = (std::int32_t) free_tiles_wrap;
                } while (free_tiles < num_tiles);
                record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
            }
    #else
            do {
                std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
                std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
                free_tiles = (std::int32_t) free_tiles_wrap;
            } while (free_tiles < num_tiles);
    #endif
        }
#else
        do {
            std::uint16_t tiles_acked = (std::uint16_t) reg_read((std::uint32_t)tiles_acked_ptr);
            std::uint16_t free_tiles_wrap = outputs[output].f.fifo_size_tiles - (outputs[output].f.tiles_received - tiles_acked);
            free_tiles = (std::int32_t) free_tiles_wrap;
        } while (free_tiles < num_tiles);
#endif
    } else {
#if defined(PERF_DUMP)
        if (wait_for_tile_en) {
    #if PERF_DUMP_LEVEL > 0
        if (regfile[p_gpr_pack::PACK_STREAM_SYNC + output] > (0)) {
            uint32_t event_id = perf::get_event_id(
                operand, num_tiles, perf::EventType::WAIT_FOR_FREE_TILES, current_outer_loop_iter);
            record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
            while (regfile[p_gpr_pack::PACK_STREAM_SYNC + output] > (0))
                ;  // Wait for prev push_tiles to complete write to register
            record_timestamp_64b(event_id, 6);  // Leave space for last-pack end-time, its possible upper 32b, and num_tiles
        }
        if constexpr (!skip_sync) {
            regfile[p_gpr_pack::PACK_STREAM_SYNC + output]++;
            sync_regfile_write(p_gpr_pack::PACK_STREAM_SYNC + output);
        }

        stream_wait_for_free_tiles(operand, get_operand_stream_id(operand), num_tiles, num_words, 0, dram_output_no_push);
    #else
        while (regfile[p_gpr_pack::PACK_STREAM_SYNC + output] > (0))
            ;  // Wait for prev push_tiles to complete write to register
        if constexpr (!skip_sync) {
            regfile[p_gpr_pack::PACK_STREAM_SYNC + output]++;
            sync_regfile_write(p_gpr_pack::PACK_STREAM_SYNC + output);
        }

        stream_wait_for_free_tiles(operand, get_operand_stream_id(operand), num_tiles, num_words, 0, dram_output_no_push);
    #endif
        }
#else
        while (regfile[p_gpr_pack::PACK_STREAM_SYNC + output] > (0))
            ;  // Wait for prev push_tiles to complete write to register
        if constexpr (!skip_sync) {
            regfile[p_gpr_pack::PACK_STREAM_SYNC + output]++;
            sync_regfile_write(p_gpr_pack::PACK_STREAM_SYNC + output);
        }

        stream_wait_for_free_tiles(operand, get_operand_stream_id(operand), num_tiles, num_words, 0, dram_output_no_push);
#endif

        if (stream_should_packer_reset_pointers(get_operand_stream_id(operand))) {
            outputs[output].f.fifo_wr_ptr = outputs[output].f.fifo_wr_base_ptr;
        }

        if (outputs[output].f.fork) {
            for (std::uint32_t k = 0; k < outputs[output].f.num_fork_streams; k++) {
#ifdef PERF_DUMP
                if (wait_for_tile_en) {
                    stream_wait_for_free_tiles(operand, outputs[output].f.fork_stream_ids[k], num_tiles, num_words, 1+k, dram_output_no_push);
                }
#else
                stream_wait_for_free_tiles(operand, outputs[output].f.fork_stream_ids[k], num_tiles, num_words, 1+k, dram_output_no_push);
#endif
            }
        }
    }

}

inline void llk_push_to_intermediate(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    std::uint32_t output = operand_to_output_index(operand);
    outputs[output].f.tiles_received += num_tiles;
#if defined(PERF_DUMP) && MATH_PACK_DECOUPLE
    volatile std::uint32_t tt_reg_ptr * tiles_received_ptr = get_operand_tiles_received_ptr(operand);
    tiles_received_ptr[0] = outputs[output].f.tiles_received & 0xffff;
#else
    volatile std::uint32_t tt_reg_ptr * tiles_received_ptr =
        (volatile std::uint32_t tt_reg_ptr *)((((volatile std::uint32_t)get_operand_tiles_received_ptr(operand)) >> 2) & 0x3ffff);

    TT_SETDMAREG(0, outputs[output].f.tiles_received, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr[0]);
#endif
}

inline void llk_push_to_brisc(const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    std::uint32_t output = operand_to_output_index(operand);
    outputs[output].f.tiles_received += num_tiles;
#if defined(PERF_DUMP) && MATH_PACK_DECOUPLE
    volatile std::uint32_t tt_reg_ptr * tiles_received_ptr = get_packer_tiles_received_ptr(operand);
    tiles_received_ptr[0] = outputs[output].f.tiles_received & 0xffff;
#else
    volatile std::uint32_t tt_reg_ptr * tiles_received_ptr =
        (volatile std::uint32_t tt_reg_ptr *)((((volatile std::uint32_t)get_packer_tiles_received_ptr(operand)) >> 2) & 0x3ffff);

    TT_SETDMAREG(0, outputs[output].f.tiles_received, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr[0]);
#endif
}

inline void llk_push_to_brisc_auto_clear(const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    std::uint32_t output = operand_to_output_index(operand);
    outputs[output].f.tiles_received += num_tiles;
}

void push_to_stream(std::uint32_t output, std::uint32_t stream_id, std::int32_t num_tiles, std::int32_t num_words, std::uint32_t fork_indx, std::uint32_t dram_output_no_push) {

#if defined(PERF_DUMP) && MATH_PACK_DECOUPLE
    if (dram_output_no_push & (1 << fork_indx)) {
        volatile std::uint32_t tt_reg_ptr * tiles_received_ptr = get_operand_tiles_received_ptr(stream_id_to_operand(stream_id));
        tiles_received_ptr[0] = outputs[output].f.tiles_received & 0xffff;
    } else {
        std::uint32_t reg_addr = STREAM_REG_ADDR(stream_id, STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX);
        std::uint32_t num_msgs_received = (num_words << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE) | num_tiles;
        reinterpret_cast<uint32_t*>(reg_addr)[0] = num_msgs_received;
    }
#else
    if (dram_output_no_push & (1 << fork_indx)) {
        volatile std::uint32_t tt_reg_ptr * tiles_received_ptr =
            (volatile std::uint32_t tt_reg_ptr *)((((volatile std::uint32_t)get_operand_tiles_received_ptr(stream_id_to_operand(stream_id))) >> 2) & 0x3ffff);
        TT_SETDMAREG(0, outputs[output].f.tiles_received, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
        TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr[0]);
    } else {
        std::uint32_t reg_addr = (STREAM_REG_ADDR(stream_id, STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX) >> 2) & 0x3ffff;
        std::uint32_t num_msgs_received = (num_words << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE) | num_tiles;
        TT_SETDMAREG(0, num_msgs_received & 0xffff, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
        TT_SETDMAREG(0, num_msgs_received >> 16, 0, HI_16(p_gpr_pack::NUM_MSGS_RECEIVED));
        TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reg_addr);
    }
#endif
} 

inline void llk_push_to_output(
    const std::int32_t operand,
    const std::int32_t stream_id,
    const std::int32_t num_tiles,
    const std::int32_t num_words) {
    std::uint32_t output = operand_to_output_index(operand);
    std::uint32_t dram_output_no_push = outputs[output].f.dram_output_no_push;
    stream_set_tiles_left_in_phase(stream_id, num_tiles);
    if (outputs[output].f.fork) {
        for (std::uint32_t k = 0; k < outputs[output].f.num_fork_streams; k++) {
            stream_set_tiles_left_in_phase(outputs[output].f.fork_stream_ids[k], num_tiles);
        }
    }

    outputs[output].f.tiles_received += num_tiles;

#if defined(PERF_DUMP) && MATH_PACK_DECOUPLE
#else
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
#endif

    push_to_stream(output, stream_id, num_tiles, num_words, 0, dram_output_no_push);

    if (outputs[output].f.fork) {
        for (std::uint32_t k = 0; k < outputs[output].f.num_fork_streams; k++) {
            push_to_stream(output, outputs[output].f.fork_stream_ids[k], num_tiles, num_words, 1+k, dram_output_no_push);
        }
    }

#if defined(PERF_DUMP) && MATH_PACK_DECOUPLE
    std::uint32_t current_stream_sync_val = regfile[p_gpr_pack::PACK_STREAM_SYNC + output];
    regfile[p_gpr_pack::PACK_STREAM_SYNC + output] = current_stream_sync_val - 1;
    sync_regfile_write(p_gpr_pack::PACK_STREAM_SYNC + output);
#else
    TT_SUBDMAREG(1, p_gpr_pack::PACK_STREAM_SYNC + output, 1, p_gpr_pack::PACK_STREAM_SYNC + output);
#endif
}


// Push N tiles to stream buffer (increment write pointer)
template <bool push_blocks = false, bool brisc_pack = false>
inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {

    std::uint32_t output = operand_to_output_index(operand);
    std::uint32_t num_words = num_tiles * (std::uint32_t)outputs[output].f.tile_size_words;
    std::uint32_t num_tiles_in_block = 0;
    bool legacy_pack = false;
    if constexpr (brisc_pack) {
       legacy_pack = outputs[output].f.legacy_pack;
    } 

    bool brisc_auto_clearing_en = false;
#if defined(PERF_DUMP)
#if OVERLAY_OUTPUT_DECOUPLE == 1
    brisc_auto_clearing_en = is_output_operand_decoupled(operand);
#endif
#endif

    std::uint32_t stream_id = get_operand_stream_id(operand);

    if constexpr (push_blocks) {
        num_tiles_in_block = outputs[output].f.block_tile_dim;
        outputs[output].f.block_tile_cnt+=num_tiles;
        if (outputs[output].f.block_tile_cnt < outputs[output].f.block_tile_dim) {
          return; //row of ublocks not ready yet
        } else {
           num_words = num_tiles_in_block * (std::uint32_t)outputs[output].f.tile_size_words;
           if (!brisc_auto_clearing_en) {
               stream_set_tiles_left_in_phase(stream_id, num_tiles_in_block);
           }
           outputs[output].f.block_tile_cnt=0;
        }
    }

    outputs[output].f.fifo_wr_ptr += num_words;
    outputs[output].f.fifo_wr_tile_ptr = 0;

    if (outputs[output].f.fifo_wr_ptr > outputs[output].f.fifo_limit) {
        outputs[output].f.fifo_wr_ptr -= outputs[output].f.fifo_size;
    }

    if (operand_is_intermediate(operand)) {
#if defined(PERF_DUMP) && DECOUPLINGS_EN
#else
        llk_push_to_intermediate(operand, num_tiles, num_words);
#endif
        if (outputs[output].f.accumulation_buffer) {
            outputs[output].f.curr_block++;
            if (outputs[output].f.curr_block == outputs[output].f.blocks_per_iter) {
               outputs[output].f.curr_iter++;
               if (outputs[output].f.curr_iter == outputs[output].f.num_iter) {
                  outputs[output].f.curr_iter=0;
                  outputs[output].f.fifo_wr_base_ptr = outputs[output].f.fifo_wr_ptr; //inc base ptr
               } else {
                outputs[output].f.fifo_wr_ptr = outputs[output].f.fifo_wr_base_ptr; //set wr prt to base ptr
               } 
               outputs[output].f.curr_block=0;
            }
        }    
    } else if (brisc_auto_clearing_en) {
        llk_push_to_brisc_auto_clear(operand, num_tiles, num_words);
    } else if (push_blocks) {
        llk_push_to_intermediate(operand, num_tiles_in_block, num_words);
    } else if (brisc_pack && !legacy_pack) {
        llk_push_to_brisc(operand, num_tiles, num_words);
    } else {
        llk_push_to_output(operand, stream_id, num_tiles, num_words);
    }

}

inline void llk_wait_for_free_blocks(const std::int32_t operand, const std::int32_t num_blocks) {
    llk_wait_for_free_tiles<false, true>(operand, num_blocks);
}

inline void llk_push_blocks(const std::int32_t operand, const std::int32_t num_blocks) {
    llk_push_tiles<true>(operand, num_blocks);
}

inline void llk_free_tiles(std::uint32_t operand, std::uint32_t num_tiles) {
    std::uint32_t output = operand_to_output_index(operand);
    if (outputs[output].f.accumulation_buffer) {

        std::uint32_t shared_output = operand_to_output_index(outputs[output].f.shared_buffer_operand);

        outputs[output].f.fifo_wr_ptr = outputs[shared_output].f.fifo_wr_ptr;

        outputs[output].f.fifo_wr_base_ptr = outputs[output].f.fifo_wr_ptr; //inc base ptr

        outputs[output].f.curr_iter = 0;
    }    
}

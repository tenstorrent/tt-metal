
#include "ckernel_perf_unpack_pack.h"
#include "stream_interface.h"

#pragma GCC diagnostic ignored "-Wunused-function"


namespace ckernel
{
extern uint32_t perf_index;
extern uint32_t perf_end;
// Perf-buffer are double buffered for spill_to_dram.
// Ncrisc will move one half to dram while trisc populates the other half.
// When INTERMED_DUMP = 0, we only dump into perf_buf_base[0].
extern volatile uint32_t tt_l1_ptr *perf_buf_base[2];
// Selects the half of perf_buffer that trisc is currently writing into.
extern uint8_t perf_buf_base_id;
extern bool record_perf_events;
extern uint16_t current_outer_loop_iter;
extern uint8_t thread_id;
extern int32_t dram_dump_req_local;
extern volatile uint tt_l1_ptr * ncrisc_ack_addr;
extern uint32_t header;

void allocate_perf_buffer() {
   std::int32_t perf_buf_base_addr;
   if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC0_BASE) {
      perf_buf_base_addr = l1_mem::address_map::UNPACK_PACK_PERF_BUF_BASE_ADDR + 0*TRISC_PERF_BUF_SIZE;
      perf_index = 2; // The first 4B value is always initialized to 0xbaddf00d.
      if constexpr (PERF_DUMP_CONCURRENT == 1 || INTERMED_DUMP == 1) {
         perf_end = TRISC_PERF_BUF_SIZE >> 3;
      } else {
         perf_end = 3;
      }
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[0];
      ncrisc_ack_addr = &EPOCH_INFO_PTR->perf_dram_copy_ack[0];
   } else if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      perf_buf_base_addr = l1_mem::address_map::MATH_PERF_BUF_BASE_ADDR;
      perf_index = 4; // The first 4 32b regs are skipped in recording math perf counters.
      perf_end = 16;

      // Initialize math_dram_dump_req_local in the beginning of epoch.
      // EPOCH_INFO_PTR->perf_dram_copy_req counters do not get reset between epochs.
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[1];
      ncrisc_ack_addr = &EPOCH_INFO_PTR->perf_dram_copy_ack[1];
   } else {
      perf_buf_base_addr = l1_mem::address_map::UNPACK_PACK_PERF_BUF_BASE_ADDR + TRISC_PERF_BUF_SIZE;
      perf_index = 2; // The first 4B value is always initialized to 0xbaddf00d.
      if constexpr (PERF_DUMP_CONCURRENT == 1 || INTERMED_DUMP == 1) {
         perf_end = TRISC_PERF_BUF_SIZE >> 3;
      } else {
         perf_end = 3;
      }
      TTI_SEMINIT(1, 0, 1 << semaphore::PACK_DONE);
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[2];
      ncrisc_ack_addr = &EPOCH_INFO_PTR->perf_dram_copy_ack[2];
   }
   // Tirsc starts dumping into the first half of the perf_buffers.
   perf_buf_base_id = 0;
   // Program the address for the first half of the perf buffer address.
   perf_buf_base[0] = reinterpret_cast<volatile uint32_t *>(perf_buf_base_addr);
   // Program the address for the second half of the perf buffer address.
   perf_buf_base[1] = reinterpret_cast<volatile uint32_t *>(perf_buf_base_addr + (TRISC_PERF_BUF_SIZE >> 1));
   perf_buf_base[perf_buf_base_id][0] = PERF_DUMP_END_SIGNAL;
#if PERF_DUMP_CONCURRENT
   volatile uint32_t* header_ptr = reinterpret_cast<volatile uint32_t *>(l1_mem::address_map::PERF_THREAD_HEADER);
   header = header_ptr[0];
   header = (header & 0xfff8ffff) | (((uint32_t)(thread_id) & 0b111) << 16);
   perf_buf_base[perf_buf_base_id][1] = header;
   for (uint i = 2; i < perf_index; i++) {
      perf_buf_base[perf_buf_base_id][i] = 0xffffffff;
   }
#else
   for (uint i = 1; i < perf_index; i++) {
      perf_buf_base[perf_buf_base_id][i] = 0xffffffff;
   }
#endif
}

void switch_perf_buffers() {

   if constexpr (INTERMED_DUMP || PERF_DUMP_CONCURRENT) {
      for (uint i = perf_index; i < perf_end; i++) {
         perf_buf_base[perf_buf_base_id][i] = 0xffffffff;
      }
      bool stalled = false;
      uint32_t timestamp_stall_start_l;
      uint32_t timestamp_stall_start_h;
      uint32_t timestamp_stall_end_l;
      uint32_t timestamp_stall_end_h;

      // Before advancing to the other half of perf-buffer, make sure ncrisc is done copying that half into dram
      int32_t ack_local = *ncrisc_ack_addr;
      if (ack_local <= dram_dump_req_local - 1) {
         stalled = true;
         timestamp_stall_start_l = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
         timestamp_stall_start_h = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);

         while (ack_local <= dram_dump_req_local - 1) {
            ack_local = *ncrisc_ack_addr;
         }

         timestamp_stall_end_l = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
         timestamp_stall_end_h = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
      }
      
      dram_dump_req_local++;
      EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] = dram_dump_req_local;

      perf_buf_base_id = 1 - perf_buf_base_id;
      if constexpr(INTERMED_DUMP) {
         perf_index = 0;
      } else {
         perf_index = 0;
         perf_buf_base[perf_buf_base_id][perf_index] = PERF_DUMP_END_SIGNAL;
         perf_buf_base[perf_buf_base_id][perf_index+1] = *(uint32_t*)(&header);
         perf_index = 2;
      }
      if (stalled && perf_index + 5 < perf_end - 1) {
         uint32_t event_id = perf::get_event_id(0, 0, perf::EventType::STALL_TRISC_FOR_DRAM_PERF_DUMP, current_outer_loop_iter);
         perf_buf_base[perf_buf_base_id][perf_index] = event_id;
         perf_buf_base[perf_buf_base_id][perf_index+1] = timestamp_stall_start_h;
         perf_buf_base[perf_buf_base_id][perf_index+2] = timestamp_stall_start_l;
         perf_buf_base[perf_buf_base_id][perf_index+3] = event_id;
         perf_buf_base[perf_buf_base_id][perf_index+4] = timestamp_stall_end_h;
         perf_buf_base[perf_buf_base_id][perf_index+5] = timestamp_stall_end_l;
         perf_index += 6;
      }
   }
}

void last_trisc_perf_dump_to_dram() {
   if (perf_index > 0) {

      // Before advancing to the other half of perf-buffer, make sure ncrisc is done copying that half into dram
      int32_t ack_local = *ncrisc_ack_addr;
      while (ack_local <= dram_dump_req_local - 1) {
         ack_local = *ncrisc_ack_addr;
      }

      if constexpr (INTERMED_DUMP) {
         if (thread_id == 1) {
            dram_dump_req_local += 2;
         } else {
            dram_dump_req_local++;
         }
      } else if constexpr (PERF_DUMP_CONCURRENT) {
         dram_dump_req_local++;
      } else {
         dram_dump_req_local += 2;
      }
      EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] = dram_dump_req_local;
   }
}

void increment_unpack_tiles(uint operand_idx, uint num_tiles) {
   if (record_perf_events && (perf_events_target_idx == 1)) {
      if (operand_idx >= PERF_MAX_NUM_INPUTS) {
         return;
      }
      uint regfile_base_idx = p_gpr_unpack::PERF_UNPACK_NUM_TILES_0;
      regfile_base_idx += (operand_idx >> 1);
      bool upper = operand_idx & 0b1;
      uint32_t num_tiles_regfile = regfile[regfile_base_idx];
      uint32_t current_num_tiles;
      if (upper) {
         current_num_tiles = (num_tiles_regfile >> 16) & 0xffff;
         current_num_tiles += num_tiles;
         regfile[regfile_base_idx] = (num_tiles_regfile & 0xffff) + ((current_num_tiles & 0xffff) << 16);
      } else {
         current_num_tiles = (num_tiles_regfile + num_tiles) & 0xffff;
         regfile[regfile_base_idx] = (num_tiles_regfile & 0xffff0000) + (current_num_tiles & 0xffff);
      }
      sync_regfile_write(regfile_base_idx);
   }   
}

void increment_pack_tiles(uint num_tiles) {
   if (record_perf_events && (perf_events_target_idx == 1)) {
      regfile[p_gpr_pack::PERF_PACK_NUM_TILES] += num_tiles;
      sync_regfile_write(p_gpr_pack::PERF_PACK_NUM_TILES);
   }
}

#if OVERLAY_DECOUPLE == 1

// This runs prior to set_perf_dump_flag_for_input so perf_end has to be adjusted
void record_overlay_decoupled_output_bw_start(uint32_t num_tiles) {
   if constexpr(PERF_DUMP_CONCURRENT == 0 && INTERMED_DUMP == 0) {
      perf_end += 6;
   }
   if (perf_end > (TRISC_PERF_BUF_SIZE >> 2)) {
      perf_end = TRISC_PERF_BUF_SIZE >> 2;
   }
   uint32_t event_id = get_event_id(0, 0, perf::EventType::OUTPUT_NUM_TILES, perf_events_target_inputs[0]);
   record_perf_value_and_check_overflow(event_id, num_tiles, 0);
   event_id = get_event_id(0, 0, perf::EventType::OUTPUT_TIMESTAMP, perf_events_target_inputs[0]);
   uint32_t timestamp_low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
   uint32_t timestamp_high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
   record_perf_value_and_check_overflow(event_id, timestamp_low, timestamp_high, 0);
}

void record_overlay_decoupled_output_bw_end() {
   if constexpr(PERF_DUMP_CONCURRENT == 0 && INTERMED_DUMP == 0) {
      perf_end += 6;
   }
   if (perf_end > (TRISC_PERF_BUF_SIZE >> 2)) {
      perf_end = TRISC_PERF_BUF_SIZE >> 2;
   }
   uint32_t event_id = get_event_id(0, 0, perf::EventType::OUTPUT_TIMESTAMP, perf_events_target_inputs[0]);
   uint32_t timestamp_low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
   uint32_t timestamp_high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
   record_perf_value_and_check_overflow(event_id, timestamp_low, timestamp_high, 0);
}

void llk_push_all_packer_tiles_for_decoupling() {
   uint32_t operand = OPERAND_OUTPUT_START_INDEX;
   uint32_t output = operand_to_output_index(operand);

   // Populate the output buffer with headers
   uint32_t stream_buf_size_bytes = EPOCH_INFO_PTR->outputs[output]->buf_full_size_bytes;
   uint32_t stream_buf_addr = EPOCH_INFO_PTR->outputs[output]->buf_base_addr;
   uint32_t stream_msg_info_buf_ptr = (EPOCH_INFO_PTR->outputs[output]->msg_info_buf_start)*MEM_WORD_WIDTH;
   uint32_t tile_size_words = *(volatile uint32_t tt_l1_ptr *)(stream_msg_info_buf_ptr);
   uint32_t tile_size_bytes = tile_size_words*MEM_WORD_WIDTH;
   for (uint32_t tile_header_ptr = stream_buf_addr; tile_header_ptr < stream_buf_addr + stream_buf_size_bytes; tile_header_ptr += tile_size_bytes) {
         *((uint32_t *)(tile_header_ptr)) = tile_size_words;
   }

   uint32_t total_num_tiles_to_push = 0;
   uint32_t num_tiles_to_push[EPOCH_MAX_OUTPUT_FORKS+1];
   uint32_t stream_id = EPOCH_INFO_PTR->outputs[output]->stream_id;
   uint32_t active_stream_idx = get_active_stream_idx(stream_id);
   volatile epoch_stream_info_t * l1_stream_info = EPOCH_INFO_PTR->active_streams[active_stream_idx];
   for (int32_t k = 0; k < l1_stream_info->num_fork_streams+1; k++) {
      uint32_t fork_active_streams_idx = k == 0 ? active_stream_idx : l1_stream_info->fork_idxs[k-1];
      uint32_t epoch_num_tiles = EPOCH_INFO_PTR->active_streams[fork_active_streams_idx]->epoch_num_tiles;
      num_tiles_to_push[k] = epoch_num_tiles;
      total_num_tiles_to_push += epoch_num_tiles;
   }
   if (((l1_stream_info->flags & STREAM_MOVES_RAW_DATA) != 0) || l1_stream_info->legacy_pack) {

      record_overlay_decoupled_output_bw_start(total_num_tiles_to_push);
      
      while(total_num_tiles_to_push > 0) {
         uint32_t stream_msg_info_buf_ptr = (l1_stream_info->msg_info_buf_start)*MEM_WORD_WIDTH;
         uint32_t tile_size_words = *(volatile uint32_t *)(stream_msg_info_buf_ptr);
         uint32_t stream_buf_size_tiles = l1_stream_info->buf_size_tiles;
         bool any_streams_busy = false;
         for (int32_t k = 0; k < l1_stream_info->num_fork_streams+1; k++) {
               uint32_t fork_active_streams_idx = k == 0 ? active_stream_idx : l1_stream_info->fork_idxs[k-1];
               uint32_t fork_stream_id = k == 0 ? stream_id : EPOCH_INFO_PTR->active_streams[fork_active_streams_idx]->stream_id;
               if (num_tiles_to_push[k] == 0) {
                  continue;
               }
               uint32_t dram_output_no_push = ((EPOCH_INFO_PTR->active_streams[fork_active_streams_idx]->flags & STREAM_DRAM_NO_PUSH) != 0) || ((EPOCH_INFO_PTR->active_streams[fork_active_streams_idx]->flags & STREAM_MOVES_RAW_DATA) != 0);
               if (dram_output_no_push) {
                  uint32_t tiles_left_in_phase = stream_src_endpoint_get_phase_tiles_count(fork_stream_id);
                  uint16_t operand_tiles_received = (uint16_t)*get_operand_tiles_received_ptr(stream_id_to_operand(fork_stream_id));
                  uint16_t operand_tiles_acked = (uint16_t)*get_operand_tiles_acked_ptr(stream_id_to_operand(fork_stream_id));
                  uint16_t tiles_available = operand_tiles_received - operand_tiles_acked;// op_pack_tiles_ptr_sub(operand_tiles_received, operand_tiles_acked);
                  uint32_t stream_buf_free_tiles = stream_buf_size_tiles - tiles_available;
                  uint32_t num_tiles = tiles_left_in_phase > stream_buf_free_tiles ? stream_buf_free_tiles : tiles_left_in_phase;
                  if (num_tiles > 0) {
                     stream_set_tiles_left_in_phase(fork_stream_id, num_tiles);
                     volatile uint32_t tt_reg_ptr* tiles_received_ptr = (volatile uint32_t tt_reg_ptr*)get_operand_tiles_received_ptr(stream_id_to_operand(fork_stream_id));
                     operand_tiles_received = (uint16_t)tiles_received_ptr[0];
                     uint16_t new_epoch_tiles_received = operand_tiles_received + num_tiles;// op_pack_tiles_ptr_add(operand_tiles_received, num_tiles);
                     tiles_received_ptr[0] = new_epoch_tiles_received;
                     
                     num_tiles_to_push[k] -= num_tiles;
                     total_num_tiles_to_push -= num_tiles;
                  }
               } else {
                  uint32_t phase_active = stream_phase_is_active(fork_stream_id) && !is_dummy_phase(fork_stream_id);
                  if (phase_active) {
                     uint32_t tiles_left_in_phase = stream_src_endpoint_get_phase_tiles_count(fork_stream_id);
                     uint32_t num_free_words = stream_get_free_words(fork_stream_id);
                     uint32_t num_tiles = 0;
                     uint32_t num_words = 0;
                     while (num_words + tile_size_words <= num_free_words && num_tiles + 1 <= tiles_left_in_phase) {
                           num_tiles++;
                           num_words += tile_size_words;
                     }
                     if (num_tiles > 0) {
                           stream_set_tiles_left_in_phase(fork_stream_id, num_tiles);
                           stream_relay_tiles(fork_stream_id, num_tiles, num_words);

                           num_tiles_to_push[k] -= num_tiles;
                           total_num_tiles_to_push -= num_tiles;
                     }
                  }
               }
         }
      }
      record_overlay_decoupled_output_bw_end();
   }
}
#endif

}

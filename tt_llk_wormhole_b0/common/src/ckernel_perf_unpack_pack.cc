
#include "ckernel_perf_unpack_pack.h"

#pragma GCC diagnostic ignored "-Wunused-function"


namespace ckernel
{
extern uint32_t perf_index;
extern uint32_t perf_end;
// Perf-buffer are double buffered for spill_to_dram.
// Ncrisc will move one half to dram while trisc populates the other half.
// When INTERMED_DUMP = 0, we only dump into perf_buf_base[0].
extern volatile uint32_t *perf_buf_base[2];
// Selects the half of perf_buffer that trisc is currently writing into.
extern uint8_t perf_buf_base_id;
extern bool record_perf_events;
extern uint16_t current_outer_loop_iter;
extern uint32_t last_clock_32h;
extern uint8_t thread_id;
extern int32_t dram_dump_req_local;
extern volatile uint* ncrisc_ack_addr;
extern uint32_t header;

void allocate_perf_buffer() {
   std::int32_t perf_buf_base_addr;
   if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC0_BASE) {
      perf_buf_base_addr = l1_mem::address_map::UNPACK_PACK_PERF_BUF_BASE_ADDR + 0*TRISC_PERF_BUF_SIZE;
      perf_index = 2; // The first 4B value is always initialized to 0xbaddf00d.
      perf_end = 3;
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
      perf_end = 3;
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
         timestamp_stall_start_l = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
         timestamp_stall_start_h = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

         while (ack_local <= dram_dump_req_local - 1) {
            ack_local = *ncrisc_ack_addr;
         }

         timestamp_stall_end_l = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
         timestamp_stall_end_h = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
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

// void record_timestamp_64b(uint event_id, uint leave_space) {
//    if (record_perf_events) {
//       uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
//       uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
//       record_perf_value_and_check_overflow(event_id, timestamp_low, timestamp_high, leave_space);
//    }
// }

}

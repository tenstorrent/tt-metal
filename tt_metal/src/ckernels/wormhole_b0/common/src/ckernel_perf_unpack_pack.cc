
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

void allocate_perf_buffer() {
   std::int32_t perf_buf_base_addr;
   if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC0_BASE) {
      perf_buf_base_addr = l1_mem::address_map::PERF_BUF_BASE_ADDR + 0*TRISC_PERF_BUF_SIZE;
      perf_index = 2; // The first 4B value is always initialized to 0xbaddf00d.
      perf_end = 3;
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[0];
   } else if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      perf_buf_base_addr = l1_mem::address_map::PERF_BUF_BASE_ADDR + 1*TRISC_PERF_BUF_SIZE;
      perf_index = 4; // The first 4 32b regs are skipped in recording math perf counters.

      // In Spill to Dram mode, we use double buffering. The size of the buffer is set to half of total l1 space allocated.
      if constexpr (INTERMED_DUMP) {
         perf_end = TRISC_PERF_BUF_SIZE >> 3;
      } else {
         perf_end = TRISC_PERF_BUF_SIZE >> 2;
      }
      // Initialize math_dram_dump_req_local in the beginning of epoch.
      // EPOCH_INFO_PTR->perf_dram_copy_req counters do not get reset between epochs.
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[1];
      // Following two gprs are used for THCON to update perf_dram_copy_req[1].
      regfile[p_gpr_math::PERF_EPOCH_BASE_ADDR] = EPOCH_INFO_ADDR >> 4;
      regfile[p_gpr_math::PERF_EPOCH_OFFSET] = uint32_t(&EPOCH_INFO_PTR->perf_dram_copy_req[1]) - (EPOCH_INFO_ADDR & 0xfffffff0);
      regfile[p_gpr_math::PERF_BUF_BASE] = perf_buf_base_addr >> 4;
   } else {
      perf_buf_base_addr = l1_mem::address_map::PERF_BUF_BASE_ADDR + 2*TRISC_PERF_BUF_SIZE;
      perf_index = 2; // The first 4B value is always initialized to 0xbaddf00d.
      perf_end = 3;
      TTI_SEMINIT(1, 0, 1 << semaphore::PACK_DONE);
      dram_dump_req_local = EPOCH_INFO_PTR->perf_dram_copy_req[2];
   }
   // Tirsc starts dumping into the first half of the perf_buffers.
   perf_buf_base_id = 0;
   // Program the address for the first half of the perf buffer address.
   perf_buf_base[0] = reinterpret_cast<volatile uint32_t *>(perf_buf_base_addr);
   // Program the address for the second half of the perf buffer address.
   perf_buf_base[1] = reinterpret_cast<volatile uint32_t *>(perf_buf_base_addr + (TRISC_PERF_BUF_SIZE >> 1));
   perf_buf_base[perf_buf_base_id][0] = PERF_DUMP_END_SIGNAL;
   for (uint i = 1; i < perf_index; i++) {
      perf_buf_base[perf_buf_base_id][i] = 0xffffffff;
   }
}

void record_perf_value_and_check_overflow(uint32_t event_id, uint32_t event_value, uint32_t leave_space) {
   // In l1 mode always reserve the last event for PERF_DUMP_END_SIGNAL.
   int reserve_space_for_trisc_end_signal = 1;
   if constexpr (INTERMED_DUMP) {
      leave_space = 0;
      reserve_space_for_trisc_end_signal = 0;
   }
   // Record a single value, and a timestamp
   if (perf_index + 1 < perf_end - reserve_space_for_trisc_end_signal - leave_space) {
      record_perf_value(event_id, event_value);
   } else {
      if constexpr (INTERMED_DUMP) {
         switch_perf_buffers_and_record_event(event_id, event_value);
      }
   }
}

void adjust_timestamp_upper_32b_and_check_overflow(uint32_t timestamp_upper_32b, uint leave_space) {
   if (timestamp_upper_32b != last_clock_32h) {
      last_clock_32h = timestamp_upper_32b;
      uint event_id_32h = perf::get_event_id(0, 0, perf::EventType::WALL_CLOCK_TOP_32B);
      record_perf_value_and_check_overflow(event_id_32h, timestamp_upper_32b, leave_space);
   }
}

// This function gets called when half-perf-buffer is full and need to switch.
// Only used for threads 0 and 2.
// For thread 1 a different function is used: switch_perf_buffers_for_math_thread
// If ncrisc has not yet finished dumping the next half of perf-buffer, trisc will stall.
// If is_perf_end_signal is true, we just need to write the PERF_DUMP_END_SIGNAL.
// This function should only get executed in INTERMED_DUMP mode.
void switch_perf_buffers_and_record_event(uint32_t event_id, uint32_t event_value, bool is_perf_end_signal) {

   // If trisc gets stalled due to ncrisc falling behind,
   // we record an event to indicate the start and end of trisc stalling.
   // Will keep the timestamps to record all of them after ncrisc catches up.
   bool trisc_stalled = false;
   uint32_t wait_for_dram_start_l;
   uint32_t wait_for_dram_start_h;
   uint32_t wait_for_dram_end_l;
   uint32_t wait_for_dram_end_h;

   if constexpr (INTERMED_DUMP) {
      int32_t ack_local = EPOCH_INFO_PTR->perf_dram_copy_ack[thread_id];
      if (ack_local <= dram_dump_req_local - 2) {
         trisc_stalled = true;
         wait_for_dram_start_l = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
         wait_for_dram_start_h = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);

         while (ack_local <= dram_dump_req_local - 2) {}

         wait_for_dram_end_l = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
         wait_for_dram_end_h = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
      }

      dram_dump_req_local++;
      EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] = dram_dump_req_local;
      perf_buf_base_id = 1 - perf_buf_base_id;
      perf_index = 0;

      if (is_perf_end_signal) {
         record_perf_dump_end();
      } else {
         record_perf_value(event_id, event_value);
      }
   } else {
      return;
   }

   if constexpr (INTERMED_DUMP) {
      if (trisc_stalled) {
         // Record the information on trisc stalling because of ncrisc not being able to keep up in dumping perf-buffers.
         // Need to check (and update if needed) the upper 32 bits of timestamp.
         uint32_t event_id_trisc_stall = perf::get_event_id(0, 0, perf::EventType::STALL_TRISC_FOR_DRAM_PERF_DUMP, current_outer_loop_iter);
         adjust_timestamp_upper_32b(wait_for_dram_start_h);
         record_perf_value(event_id_trisc_stall, wait_for_dram_start_l);
         adjust_timestamp_upper_32b(wait_for_dram_end_h);
         record_perf_value(event_id_trisc_stall, wait_for_dram_end_l);
      }
   }
}

void record_timestamp_64b(uint event_id, uint leave_space) {
   if (record_perf_events) {
      uint32_t timestamp_low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
      uint32_t timestamp_high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
      adjust_timestamp_upper_32b_and_check_overflow(timestamp_high, leave_space);
      record_perf_value_and_check_overflow(event_id, timestamp_low, leave_space);
   }
}

void record_perf_dump_end_and_check_overflow() {
   if (perf_index < perf_end) {
      record_perf_dump_end();
   } else {
      if constexpr (INTERMED_DUMP) {
         switch_perf_buffers_and_record_event(0, 0, true);
      }
   }
}

}

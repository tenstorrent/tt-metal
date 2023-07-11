#pragma once

#include <cstdint>
#include <l1_address_map.h>
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "tensix.h"
#include "fw_debug.h"
#include "epoch.h"

#include "ckernel_perf_include.h"

#pragma GCC diagnostic ignored "-Wunused-function"

// Comment in/out to enable perf scratch even logging

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
extern uint32_t last_clock_32h;
extern uint8_t thread_id;

// In math thread, THCON dumps perf buffers in l1.
// Therefore, incrementing the ncrisc perf_dram_buffer_req must be done by THCON as well.
// Flipping the l1 perf start address must also be done by THCON for math thread.
// Following variable keeps track of latest value of perf_dram_copy_req[1] from trisc perspective.
// The actual value might be different, because the queued THCON updates for perf_dram_copy_req[1] might have yet not been executed.
// We read this value initially for all threads to reduce the l1-reads.
extern int32_t dram_dump_req_local;
extern bool record_perf_events;
extern bool first_unpack_recorded;
extern volatile uint tt_l1_ptr * ncrisc_ack_addr;
extern uint16_t current_outer_loop_iter;

void allocate_perf_buffer();

// This function gets called when half-perf-buffer is full and need to switch.
// Only used for threads 0 and 2.
// For thread 1 a different function is used: switch_perf_buffers_for_math_thread
// If ncrisc has not yet finished dumping the next half of perf-buffer, trisc will stall.
// If is_perf_end_signal is true, we just need to write the PERF_DUMP_END_SIGNAL.
// This function should only get executed in INTERMED_DUMP mode.
void switch_perf_buffers();
void last_trisc_perf_dump_to_dram();

// The two following functions are separated to avoid inline recursive function calls.
// TODO: Check the behaviour of the compiler if the two following functions were merged into a template function.
inline void record_perf_value(uint32_t event_id, uint32_t event_value_lo_32b, uint32_t event_value_hi_32b) {
   // In l1 mode always reserve the last event for PERF_DUMP_END_SIGNAL.
   int reserve_space_for_trisc_end_signal = 1;
   if constexpr (INTERMED_DUMP || PERF_DUMP_CONCURRENT) {
      reserve_space_for_trisc_end_signal = 0;
   }
   FWASSERT("There is no space left in perf-buffer.", perf_index + 1 < perf_end - reserve_space_for_trisc_end_signal);
   perf_buf_base[perf_buf_base_id][perf_index] = event_id;
   perf_buf_base[perf_buf_base_id][perf_index + 1] = event_value_hi_32b;
   perf_buf_base[perf_buf_base_id][perf_index + 2] = event_value_lo_32b;
   perf_index += 3;
}

inline void record_perf_dump_end() {
   if (perf_index < perf_end) {
      perf_buf_base[perf_buf_base_id][perf_index] = PERF_DUMP_END_SIGNAL;
      perf_index += 1;      
   }
#if PERF_DUMP_CONCURRENT == 1
   if (perf_index < perf_end) {
      perf_buf_base[perf_buf_base_id][perf_end - 1] = PERF_DUMP_END_SIGNAL;
   }
#endif
}

inline void record_perf_value_and_check_overflow(uint32_t event_id, uint32_t event_value_lo_32b, uint32_t event_value_hi_32b, uint32_t leave_space = 0) {
   // In l1 mode always reserve the last event for PERF_DUMP_END_SIGNAL.
   int reserve_space_for_trisc_end_signal = 1;

#if (INTERMED_DUMP == 1) || (PERF_DUMP_CONCURRENT == 1)
   leave_space = 0;
   reserve_space_for_trisc_end_signal = 0;
   if (perf_index + 2 >= perf_end - reserve_space_for_trisc_end_signal - leave_space) {
      switch_perf_buffers();
   }
   record_perf_value(event_id, event_value_lo_32b, event_value_hi_32b);
#else
   if (perf_index + 2 < perf_end - reserve_space_for_trisc_end_signal - leave_space) {
      record_perf_value(event_id, event_value_lo_32b, event_value_hi_32b);
   }
#endif
}

inline void record_timestamp_64b(uint event_id, uint leave_space = 0) {
   if (record_perf_events) {
      uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
      uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
      record_perf_value_and_check_overflow(event_id, timestamp_low, timestamp_high, leave_space);
   }
}

inline void record_perf_dump_end_and_check_overflow() {
   if (thread_id == 1) {
      uint32_t reserve_space_for_trisc_end_signal = 1;
      if (perf_index + 3 <= perf_end-reserve_space_for_trisc_end_signal) { // Last event is always set to a default.
         perf_buf_base[perf_buf_base_id][perf_index] = reg_read(0xFFB12000 + 0x120);
         perf_buf_base[perf_buf_base_id][perf_index+1] = reg_read(0xFFB12000 + 0x124);
         perf_buf_base[perf_buf_base_id][perf_index+2] = 0;
         perf_buf_base[perf_buf_base_id][perf_index+3] = 0;
         perf_index += (PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t));
      }
   }
#if (INTERMED_DUMP == 1) || (PERF_DUMP_CONCURRENT == 1)
   if (perf_index >= perf_end) {
      switch_perf_buffers();
   }
   record_perf_dump_end();
#else
   if (perf_index < perf_end) {
      record_perf_dump_end();
   }
#endif
}

inline void record_latest_wait_for_tile() {
#if defined(PERF_DUMP)
   if (!first_unpack_recorded) {
      uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
      uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
      regfile[p_gpr_unpack::PERF_FIRST_UNP_LO] = timestamp_low & 0xffffffff;
      sync_regfile_write(p_gpr_unpack::PERF_FIRST_UNP_LO);
      regfile[p_gpr_unpack::PERF_FIRST_UNP_HI] = timestamp_high & 0xffffffff;
      sync_regfile_write(p_gpr_unpack::PERF_FIRST_UNP_HI);
   }
#endif
}

}

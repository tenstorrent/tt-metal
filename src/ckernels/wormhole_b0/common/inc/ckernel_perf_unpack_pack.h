#pragma once

#include <cstdint>
#include <l1_address_map.h>
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "tensix.h"
#include "fw_debug.h"

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
extern volatile uint32_t *perf_buf_base[2];
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

void allocate_perf_buffer();
// This function gets called when half-perf-buffer is full and need to switch.
// Only used for threads 0 and 2.
// For thread 1 a different function is used: switch_perf_buffers_for_math_thread
// If ncrisc has not yet finished dumping the next half of perf-buffer, trisc will stall.
// If is_perf_end_signal is true, we just need to write the PERF_DUMP_END_SIGNAL.
// This function should only get executed in INTERMED_DUMP mode.
void switch_perf_buffers_and_record_event(uint32_t event_id, uint32_t event_value, bool is_perf_end_signal = false);
void record_perf_value_and_check_overflow(uint32_t event_id, uint32_t event_value, uint32_t leave_space = 0);
void adjust_timestamp_upper_32b_and_check_overflow(uint32_t timestamp_upper_32b, uint leave_space = 0);
void record_timestamp_64b(uint event_id, uint leave_space = 0);
void record_perf_dump_end_and_check_overflow();

inline uint32_t read_wall_clock_l()
{
    volatile uint32_t * clock_lo = reinterpret_cast<volatile uint32_t * >(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return clock_lo[0]; // latches high
}

inline uint32_t read_wall_clock_h()
{
    volatile uint32_t * clock_hi = reinterpret_cast<volatile uint32_t * >(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return clock_hi[0];
}

inline uint64_t read_wall_clock()
{
    uint32_t low = read_wall_clock_l();
    uint32_t high = read_wall_clock_h();
    return ((uint64_t)high << 32) | low;
}

// The two following functions are separated to avoid inline recursive function calls.
// TODO: Check the behaviour of the compiler if the two following functions were merged into a template function.
inline void record_perf_value(uint32_t event_id, uint32_t event_value) {
   // In l1 mode always reserve the last event for PERF_DUMP_END_SIGNAL.
   int reserve_space_for_trisc_end_signal = 1;
   if constexpr (INTERMED_DUMP) {
      reserve_space_for_trisc_end_signal = 0;
   }
   FWASSERT("There is no space left in perf-buffer.", perf_index + 1 < perf_end - reserve_space_for_trisc_end_signal);
   perf_buf_base[perf_buf_base_id][perf_index] = event_id;
   perf_buf_base[perf_buf_base_id][perf_index + 1] = event_value;
   perf_index += 2;
}

// The two following functions are separated to avoid inline recursive function calls.
// TODO: Check the behaviour of the compiler if the two following functions were merged into a template function.
inline void adjust_timestamp_upper_32b(uint32_t timestamp_upper_32b) {
   if (timestamp_upper_32b != last_clock_32h) {
      last_clock_32h = timestamp_upper_32b;
      uint event_id_32h = perf::get_event_id(0, 0, perf::EventType::WALL_CLOCK_TOP_32B);
      record_perf_value(event_id_32h, timestamp_upper_32b);
   }
}

inline void record_perf_dump_end() {
   if (perf_index < perf_end) {
      perf_buf_base[perf_buf_base_id][perf_index] = PERF_DUMP_END_SIGNAL;
      perf_index += 1;
   }
}

inline void last_trisc_perf_dump_to_dram() {
   if (perf_index > 0) {

      int32_t ack_local = EPOCH_INFO_PTR->perf_dram_copy_ack[thread_id];
      if (ack_local <= dram_dump_req_local - 2) {
         while (ack_local <= dram_dump_req_local - 2) {}
      }

      if constexpr (INTERMED_DUMP) {
         dram_dump_req_local++;
      } else {
         dram_dump_req_local += 2;
      }
      EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] = dram_dump_req_local;
   }
}

inline void record_timestamp(uint32_t event_id, uint leave_space = 0) {
  record_perf_value_and_check_overflow(event_id, reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L));
}

}

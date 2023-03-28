#pragma once

#include <cstdint>
#include <l1_address_map.h>
#include "tensix.h"
#include "fw_debug.h"

#ifdef PERF_DUMP
#include "perf_events_target_inputs.h"
#include "perf_lib/scratch_api.h"
#endif
#include "perf_res_decouple.h"

#ifndef INTERMED_DUMP
#define INTERMED_DUMP 0
#endif

#pragma GCC diagnostic ignored "-Wunused-function"

// Comment in/out to enable perf scratch even logging

namespace ckernel
{
constexpr uint32_t PERF_DUMP_END_SIGNAL = 0xbeeff00d;
constexpr uint32_t PERF_CNT_DUMP_ENTRY_SIZE = 16; // Entry size in bytes
extern uint32_t perf_index;
extern uint32_t perf_end;
// Perf-buffer are double buffered for spill_to_dram.
// Ncrisc will move one half to dram while trisc populates the other half.
// When INTERMED_DUMP = 0, we only dump into perf_buf_base[0].
extern volatile uint32_t *perf_buf_base[2];
// Selects the half of perf_buffer that trisc is currently writing into.
extern uint8_t perf_buf_base_id;
extern bool record_perf_events;
extern uint8_t num_unpack_first_block_recorded;
extern uint8_t max_unpack_first_block_to_record;
extern uint8_t perf_events_target_idx;
extern uint16_t current_outer_loop_iter;
extern uint8_t mailbox_index;
extern uint8_t mailbox_end;
extern uint32_t last_clock_32h;
extern uint16_t perf_total_num_tiles[2];
extern uint8_t thread_id;

// In math thread, THCON dumps perf buffers in l1.
// Therefore, incrementing the ncrisc perf_dram_buffer_req must be done by THCON as well.
// Flipping the l1 perf start address must also be done by THCON for math thread.
// Following variable keeps track of latest value of perf_dram_copy_req[1] from trisc perspective.
// The actual value might be different, because the queued THCON updates for perf_dram_copy_req[1] might have yet not been executed.
// We read this value initially for all threads to reduce the l1-reads.
extern int32_t dram_dump_req_local;

static volatile uint32_t *debug_mailbox() {
  extern volatile std::uint32_t DEBUG_MAILBOX[];
  return DEBUG_MAILBOX;
}

#if PERF_DUMP_LEVEL == 0
static constexpr int32_t TRISC_PERF_BUF_SIZE = l1_mem::address_map::TRISC_PERF_BUF_SIZE_LEVEL_0;
#else
static constexpr int32_t TRISC_PERF_BUF_SIZE = l1_mem::address_map::TRISC_PERF_BUF_SIZE_LEVEL_1;
#endif

struct cperf_cnt_mode
{
    constexpr static uint32_t PERF_CNT_MODE_FREE = 0; // Free running period counter
    constexpr static uint32_t PERF_CNT_MODE_STOP = 1; // Stop counter
    constexpr static uint32_t PERF_CNT_MODE_WRAP = 2; // Wrap period counter
};

struct cperf_cnt_block_sel
{
    constexpr static uint32_t PERF_CNT_INSTR_THREAD = 0; // Select all instruction thread perf counters(includes TDMA)
    constexpr static uint32_t PERF_CNT_FPU = 1; // Select FPU perf counters
    constexpr static uint32_t PERF_CNT_L1  = 2; // Select L1 perf counters
    constexpr static uint32_t PERF_CNT_ALL = 3; // Select all perf counters
};

struct cperf_dbg_daisy_id
{
    constexpr static uint32_t DEBUG_DAISY_INSTRN_THREAD = 1; // Thread specific perf counters
    constexpr static uint32_t DEBUG_DAISY_INSTRN_ISSUE_0 = 4; // TDMA+math
    constexpr static uint32_t DEBUG_DAISY_INSTRN_ISSUE_1 = 5; // math+instruction issue
    constexpr static uint32_t DEBUG_DAISY_TENSIX  = 7; // FPU and L1 perf counters
};

struct cperf_dbg_dump_to_mem_mode
{
    constexpr static uint32_t DEBUG_MEM_MODE_MANUAL_WR = 0;
    constexpr static uint32_t DEBUG_MEM_MODE_AUTO_WR = 1;
    constexpr static uint32_t DEBUG_MEM_MODE_MANUAL_RD = 2;
    constexpr static uint32_t DEBUG_MEM_MODE_AUTO_RD = 3;
};

inline void allocate_perf_buffer() {
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

// This function gets called when half-perf-buffer is full and need to switch.
// Only used for threads 0 and 2.
// For thread 1 a different function is used: switch_perf_buffers_for_math_thread
// If ncrisc has not yet finished dumping the next half of perf-buffer, trisc will stall.
// If is_perf_end_signal is true, we just need to write the PERF_DUMP_END_SIGNAL.
// This function should only get executed in INTERMED_DUMP mode.
inline void switch_perf_buffers_and_record_event(uint32_t event_id, uint32_t event_value, bool is_perf_end_signal = false) {

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
      // perf_buf_base[perf_buf_base_id][perf_index] = 0xabcde0;
      // perf_buf_base[perf_buf_base_id][perf_index+1] = EPOCH_INFO_PTR->perf_dram_copy_req[thread_id];
      // perf_index += 2;
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
   // perf_buf_base[perf_buf_base_id][perf_index] = 0xabcde1;
   // perf_buf_base[perf_buf_base_id][perf_index+1] = EPOCH_INFO_PTR->perf_dram_copy_req[thread_id];
   // perf_index += 2;
   }
}

inline void record_perf_value_and_check_overflow(uint32_t event_id, uint32_t event_value, uint32_t leave_space = 0) {
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

inline void adjust_timestamp_upper_32b_and_check_overflow(uint32_t timestamp_upper_32b, uint leave_space = 0) {
   if (timestamp_upper_32b != last_clock_32h) {
      last_clock_32h = timestamp_upper_32b;
      uint event_id_32h = perf::get_event_id(0, 0, perf::EventType::WALL_CLOCK_TOP_32B);
      record_perf_value_and_check_overflow(event_id_32h, timestamp_upper_32b, leave_space);
   }
}

inline void record_timestamp(uint32_t event_id, uint leave_space = 0) {
  record_perf_value_and_check_overflow(event_id, reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L));
}

inline void record_timestamp_64b(uint event_id, uint leave_space = 0) {
   uint32_t timestamp_low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
   uint32_t timestamp_high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
   adjust_timestamp_upper_32b_and_check_overflow(timestamp_high, leave_space);
   record_perf_value_and_check_overflow(event_id, timestamp_low, leave_space);
}

inline void set_perf_cnt_params(uint32_t block_sel=cperf_cnt_block_sel::PERF_CNT_FPU, uint32_t ref_period=0xffffffff, uint32_t mode=cperf_cnt_mode::PERF_CNT_MODE_FREE) {
  uint32_t perf_cnt_ref_period_reg;
  switch (block_sel) {
     case cperf_cnt_block_sel::PERF_CNT_INSTR_THREAD:     perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0; break;
     case cperf_cnt_block_sel::PERF_CNT_L1:              perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
     default: perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_FPU0;
  }
  reg_write(perf_cnt_ref_period_reg, ref_period);
  reg_write(perf_cnt_ref_period_reg+4, mode);
}

inline void stop_perf_cnt(uint32_t block_sel=cperf_cnt_block_sel::PERF_CNT_FPU) {
  uint32_t perf_cnt_cntl_reg;
  switch (block_sel) {
     case cperf_cnt_block_sel::PERF_CNT_INSTR_THREAD: perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2; break;
     case cperf_cnt_block_sel::PERF_CNT_L1:           perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_L1_2; break;
     case cperf_cnt_block_sel::PERF_CNT_ALL:          perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_ALL; break;
     default: perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_FPU2;
  }
  reg_write(perf_cnt_cntl_reg, 0x00000002);
  reg_write(perf_cnt_cntl_reg, 0x00000000);
}

template <bool use_tensix=true, bool stall_on_math=false>
inline void stop_fpu_perf_cnt() {
  if constexpr (use_tensix) {
      if constexpr (stall_on_math) {
         TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
      }
      TTI_STOREREG(p_gpr_math::PERF_CNT_STOP, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
      TTI_STOREREG(p_gpr::ZERO, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
  } else {
     reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000002);
     reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000000);
  }
}

inline void start_perf_cnt(uint32_t block_sel=cperf_cnt_block_sel::PERF_CNT_FPU) {
  uint32_t perf_cnt_cntl_reg;
  switch (block_sel) {
     case cperf_cnt_block_sel::PERF_CNT_INSTR_THREAD: perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2; break;
     case cperf_cnt_block_sel::PERF_CNT_L1:           perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_L1_2; break;
     case cperf_cnt_block_sel::PERF_CNT_ALL:          perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_ALL; break;
     default: perf_cnt_cntl_reg = RISCV_DEBUG_REG_PERF_CNT_FPU2;
  }
  reg_write(perf_cnt_cntl_reg, 0x00000001);
  reg_write(perf_cnt_cntl_reg, 0x00000000);
}

template <bool use_tensix=true, uint32_t wait_res=p_stall::SRCA_VLD>
inline void start_fpu_perf_cnt() {
  if constexpr (use_tensix) {
      TTI_STALLWAIT(p_stall::STALL_THCON, wait_res);
      TTI_STOREREG(p_gpr_math::PERF_CNT_START, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
      TTI_STOREREG(p_gpr::ZERO, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
  } else {
     reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000001);
     reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000000);
  }
}


inline void sel_fpu_perf_cnt(uint32_t cnt_id) {
   riscv_debug_reg_dbg_dbus_cntl_u dbg_bus_cntl;
   dbg_bus_cntl.val = reg_read_barrier(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);;
   dbg_bus_cntl.f.dbg_daisy_sel = cperf_dbg_daisy_id::DEBUG_DAISY_TENSIX;
   dbg_bus_cntl.f.dbg_sig_sel   = 0x0;
   dbg_bus_cntl.f.dbg_rd_sel    = cnt_id<<1; //rd_sel is aligned to 16-bit
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

// Return value of the selected perf counter
inline uint32_t get_perf_cnt() {
   return reg_read_barrier(RISCV_DEBUG_REG_DBG_RD_DATA);
}

template <bool use_tensix=true>
inline void dump_perf_cnt_to_mem() {
   if constexpr (use_tensix) {
      TTI_STOREREG(p_gpr_math::PERF_MEM_DUMP_CNTL_SET,   (RISCV_DEBUG_REG_DBG_L1_MEM_REG2>>2)&0x3ffff);
      TTI_STOREREG(p_gpr_math::PERF_MEM_DUMP_CNTL_CLEAR, (RISCV_DEBUG_REG_DBG_L1_MEM_REG2>>2)&0x3ffff);
   } else {
      riscv_debug_reg_dbg_l1_mem_reg2_u dbg_l1_mem_reg2;
      dbg_l1_mem_reg2.val = 0;
      dbg_l1_mem_reg2.f.mem_dump_mode = cperf_dbg_dump_to_mem_mode::DEBUG_MEM_MODE_AUTO_WR;
      dbg_l1_mem_reg2.f.mem_write = 1;
      reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG2, dbg_l1_mem_reg2.val);
      dbg_l1_mem_reg2.f.mem_write = 0;
      reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG2, dbg_l1_mem_reg2.val);
   }
}

inline void dbg_daisy_enable() {
   riscv_debug_reg_dbg_dbus_cntl_u dbg_bus_cntl;
   dbg_bus_cntl.val = reg_read_barrier(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);
   dbg_bus_cntl.f.dbg_reg_ovrd_en = 0x1;
   dbg_bus_cntl.f.dbg_daisy_en = 0x1;
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

inline void dbg_daisy_disable() {
   riscv_debug_reg_dbg_dbus_cntl_u dbg_bus_cntl;
   dbg_bus_cntl.val = reg_read_barrier(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);
   dbg_bus_cntl.f.dbg_reg_ovrd_en = 0x0;
   dbg_bus_cntl.f.dbg_daisy_en = 0x0;
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

inline void dbg_enable_dump_to_mem(uint32_t start_addr, uint32_t end_addr) {

   TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
   uint16_t start_addr_lo = (start_addr >> 4) & 0xffff;
   uint16_t start_addr_hi = (start_addr >> 4) >> 16;
   TT_SETDMAREG(0, start_addr_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, start_addr_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG0 >> 2) & 0x3ffff);

   uint16_t end_addr_lo = (end_addr >> 4) & 0xffff;
   uint16_t end_addr_hi = (end_addr >> 4) >> 16;
   TT_SETDMAREG(0, end_addr_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, end_addr_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG1 >> 2) & 0x3ffff);

   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG0, start_addr>>4);
   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG1, end_addr>>4);
   riscv_debug_reg_dbg_l1_mem_reg2_u dbg_l1_mem_reg2;
   dbg_l1_mem_reg2.val = 0;
   dbg_l1_mem_reg2.f.mem_dump_mode = 0xf; //invalid and overriden below to trigger pulse needed to latch start address
   dbg_l1_mem_reg2.f.skip_cycles = 0;

   uint16_t debug_l1_reg2_lo = dbg_l1_mem_reg2.val & 0xffff;
   uint16_t debug_l1_reg2_hi = (dbg_l1_mem_reg2.val >> 16) & 0xffff;
   TT_SETDMAREG(0, debug_l1_reg2_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, debug_l1_reg2_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG2 >> 2) & 0x3ffff);


   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG2, dbg_l1_mem_reg2.val);
   dbg_l1_mem_reg2.f.mem_dump_mode = cperf_dbg_dump_to_mem_mode::DEBUG_MEM_MODE_AUTO_WR; // This value must change in order to latch new start address!!!
   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG2, dbg_l1_mem_reg2.val);

   debug_l1_reg2_lo = dbg_l1_mem_reg2.val & 0xffff;
   debug_l1_reg2_hi = (dbg_l1_mem_reg2.val >> 16) & 0xffff;
   TT_SETDMAREG(0, debug_l1_reg2_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, debug_l1_reg2_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG2 >> 2) & 0x3ffff);

   TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::THCON);
}

inline void record_fpu_perf_event_id() {
   const uint event_id = perf::get_event_id(0, 0, perf::EventType::MATH_PERF_COUNTERS, current_outer_loop_iter);
   // In each 16B, the first 8 Bytes are the perf_counters, the third 4B regs will record the event_id.
   const uint addr_offset = (perf_index + 2) << 2;
   TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
   TT_SETDMAREG(0, event_id & 0xffff, 0, LO_16(p_gpr_math::PERF_EVENT_ID));
   TT_SETDMAREG(0, (event_id >> 16) & 0xffff, 0, HI_16(p_gpr_math::PERF_EVENT_ID));
   TT_SETDMAREG(0, addr_offset & 0xffff, 0, LO_16(p_gpr_math::PERF_EVENT_OFFSET));
   TT_SETDMAREG(0, (addr_offset >> 16) & 0xffff, 0, HI_16(p_gpr_math::PERF_EVENT_OFFSET));
   TTI_STOREIND(1, 1, 0, LO_16(p_gpr_math::PERF_EVENT_OFFSET), p_ind::INC_NONE, p_gpr_math::PERF_EVENT_ID, p_gpr_math::PERF_BUF_BASE);
   TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::THCON);
}

// In math thread THCON writes perf buffers in l1.
// The perf-buffer switch must be executed by THCON as well. To ensure correct synchronization.
// The EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] is done using STOREIND instruction.
// math_dram_dump_req_local keeps track of the latest value for EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] from trisc perspective.
// The actual value of EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] depends on how many of the STOREIND instructions that are issued in this function are executed so far.
// Also issues fpu-counter-dump start addr update. Which is also done useing THCON.
// This function should only get executed in INTERMED_DUMP mode.
inline void switch_perf_buffers_for_math_thread() {
   if constexpr (INTERMED_DUMP) {
      // if (EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] <= math_dram_dump_req_local - 2) {
      //    while (EPOCH_INFO_PTR->perf_dram_copy_req[thread_id] <= math_dram_dump_req_local - 2) {}
      // }
      int32_t ack_local = EPOCH_INFO_PTR->perf_dram_copy_ack[thread_id];
      if (ack_local <= dram_dump_req_local - 2) {
         while (ack_local <= dram_dump_req_local - 2) {}
      }

      dram_dump_req_local++;
      uint16_t dram_req_lo = dram_dump_req_local & 0xffff;
      uint16_t dram_req_hi = (dram_dump_req_local >> 16) & 0xffff;
      TT_SETDMAREG(0, dram_req_lo, 0, LO_16(p_gpr_math::NUM_DRAM_REQS));
      TT_SETDMAREG(0, dram_req_hi, 0, HI_16(p_gpr_math::NUM_DRAM_REQS));
      TTI_STOREIND(1, 1, 0, LO_16(p_gpr_math::PERF_EPOCH_OFFSET), p_ind::INC_NONE, p_gpr_math::NUM_DRAM_REQS, p_gpr_math::PERF_EPOCH_BASE_ADDR);

      perf_buf_base_id = 1 - perf_buf_base_id;
      perf_index = 0;
      dbg_enable_dump_to_mem((uint32_t)&perf_buf_base[perf_buf_base_id][0], (uint32_t)&perf_buf_base[perf_buf_base_id][perf_end]);
      int32_t perf_buf_base = l1_mem::address_map::PERF_BUF_BASE_ADDR + TRISC_PERF_BUF_SIZE;
      if (perf_buf_base_id == 1) {
         perf_buf_base += (TRISC_PERF_BUF_SIZE >> 1);
      }
      perf_buf_base = perf_buf_base >> 4;
      TT_SETDMAREG(0, perf_buf_base & 0xffff, 0, LO_16(p_gpr_math::PERF_BUF_BASE));
      TT_SETDMAREG(0, (perf_buf_base >> 16) & 0xffff, 0, HI_16(p_gpr_math::PERF_BUF_BASE));
   } else {
      return;
   }
}

template <bool use_tensix=true>
inline void record_fpu_perf_cnt_value() {
   // In l1 mode always reserve last event for PERF_DUMP_END_SIGNAL.
   uint32_t reserve_space_for_trisc_end_signal = 1;
   if constexpr (INTERMED_DUMP) {
      reserve_space_for_trisc_end_signal = 0;
   }
   if (perf_index + 3 < perf_end-reserve_space_for_trisc_end_signal) { // Last event is always set to a default.
      //perf_buf_base[perf_index] = get_perf_cnt();
      //perf_buf_base[perf_index + 1] = get_perf_cnt();
      dump_perf_cnt_to_mem<use_tensix>(); //Dump 16B to L1
      record_fpu_perf_event_id();
      perf_index += (PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t));
   } else {
      if constexpr (INTERMED_DUMP) {
         switch_perf_buffers_for_math_thread();
         dump_perf_cnt_to_mem<use_tensix>(); //Dump 16B to L1
         record_fpu_perf_event_id();
         perf_index += (PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t));
      }
   }
}

inline void record_perf_dump_end_and_check_overflow() {
   if (perf_index < perf_end) {
      record_perf_dump_end();
   } else {
      if constexpr (INTERMED_DUMP) {
         switch_perf_buffers_and_record_event(0, 0, true);
      }
   }
}

// Dump a dummy math event to get the initial fpu counter value.
inline void record_dummy_math_event() {
   if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      record_fpu_perf_cnt_value();
   }
}

inline void setup_fpu_perf_cnt() {
   // Only program perf counters for math thread (trisc1)
   if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      set_perf_cnt_params(cperf_cnt_block_sel::PERF_CNT_FPU,0xffffffff,cperf_cnt_mode::PERF_CNT_MODE_FREE);
      sel_fpu_perf_cnt(0);
      dbg_daisy_enable();
      dbg_enable_dump_to_mem((uint32_t)&perf_buf_base[perf_buf_base_id][PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t)], (uint32_t)&perf_buf_base[perf_buf_base_id][perf_end]);

      riscv_debug_reg_dbg_l1_mem_reg2_u dbg_l1_mem_reg2;
      dbg_l1_mem_reg2.val = 0;
      dbg_l1_mem_reg2.f.mem_dump_mode = cperf_dbg_dump_to_mem_mode::DEBUG_MEM_MODE_AUTO_WR;
      dbg_l1_mem_reg2.f.mem_write = 0;
      regfile[p_gpr_math::PERF_MEM_DUMP_CNTL_CLEAR]=dbg_l1_mem_reg2.val;
      dbg_l1_mem_reg2.f.mem_write = 1;
      regfile[p_gpr_math::PERF_MEM_DUMP_CNTL_SET]=dbg_l1_mem_reg2.val;

      regfile[p_gpr_math::PERF_CNT_START]=0x1;
      regfile[p_gpr_math::PERF_CNT_STOP]=0x2;
      sync_regfile_write(p_gpr_math::PERF_CNT_STOP);
   }
}

inline void set_perf_dump_flag_for_input(int input_idx) {
   perf_total_num_tiles[0] = 0;
   perf_total_num_tiles[1] = 0;
   if (perf_events_target_inputs[perf_events_target_idx] == input_idx) {
      record_perf_events = true;
      perf_events_target_idx++;
      // perf_end for math thread does not change based on number of inputs.
      // We always record one 16B event for each input.
      if (thread_id == 0 || thread_id == 2) {
         if constexpr (INTERMED_DUMP == 1) {
            perf_end = TRISC_PERF_BUF_SIZE >> 3;
         } else {
            perf_end += num_events_per_input;
            // The buffer size available for each thread after double buffering is (l1_mem::address_map::TRISC_PERF_BUF_SIZE)/2.
            // Max number of events we can record in each half of the buffer will be that size divided by 4, since each event will be 4 bytes.
            if (perf_end > (TRISC_PERF_BUF_SIZE >> 2)) {
               perf_end = TRISC_PERF_BUF_SIZE >> 2;
            }
         }
      }
      current_outer_loop_iter = input_idx;
   } else {
      record_perf_events = false;
   }
}

inline void store_latest_unpack_block_available() {
   if (record_perf_events && thread_id == 0) {
      uint32_t timestamp_low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
      uint32_t timestamp_high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
      regfile[p_gpr_unpack::PERF_LAST_WAIT_LO] = timestamp_low;
      regfile[p_gpr_unpack::PERF_LAST_WAIT_HI] = timestamp_high;
   }
}

inline void unpack_last_wait_end_time_and_num_tiles() {
   if (record_perf_events) {
      uint32_t clock_high = regfile[p_gpr_unpack::PERF_LAST_WAIT_HI];
      adjust_timestamp_upper_32b_and_check_overflow(clock_high);

      uint32_t event_id_last_wait_tile = perf::get_event_id(0, 0, perf::EventType::UNPACK_LAST_WAIT_FOR_TILE, current_outer_loop_iter);
      uint32_t clock_lo = regfile[p_gpr_unpack::PERF_LAST_WAIT_LO];
      record_perf_value_and_check_overflow(event_id_last_wait_tile, clock_lo);

      uint32_t event_id_num_tiles_op0 = perf::get_event_id(0, 0, perf::EventType::NUM_TILES, current_outer_loop_iter);
      record_perf_value_and_check_overflow(event_id_num_tiles_op0, perf_total_num_tiles[0]);
      // FIXME: Unpacker for reduce, will set "max_unpack_first_block_to_record" because hlkc passes SRCAB as wait_res.
      // However the weights are not streamed in for reduce. HLKC needs to diiferentiate between active SRCA/B and actual num-inputs for perf api.
      if (max_unpack_first_block_to_record > 1 && perf_total_num_tiles[1] > 0) {
         uint32_t event_id_num_tiles_op1 = perf::get_event_id(1, 0, perf::EventType::NUM_TILES, current_outer_loop_iter);
         record_perf_value_and_check_overflow(event_id_num_tiles_op1, perf_total_num_tiles[1]);
      }
   }
}

inline void pack_record_input_init_time() {
   if (record_perf_events) {
      t6_semaphore_post(semaphore::PACK_DONE);
      while (semaphore_read(semaphore::PACK_DONE) == 0) {}
      uint32_t event_id = perf::get_event_id(0, 0, perf::EventType::PACK_EACH_INPUT, current_outer_loop_iter);
      record_timestamp_64b(event_id);
   }
}

inline void pack_record_input_end_time_and_num_tiles() {
   if (record_perf_events) {
      t6_semaphore_get<p_stall::PACK>(semaphore::PACK_DONE);
      while (semaphore_read(semaphore::PACK_DONE) > 0) {}
      uint32_t event_id = perf::get_event_id(0, 0, perf::EventType::PACK_EACH_INPUT, current_outer_loop_iter);
      record_timestamp_64b(event_id);

      uint32_t event_id_num_tiles = perf::get_event_id(0, 0, perf::EventType::NUM_TILES, current_outer_loop_iter);
      record_perf_value_and_check_overflow(event_id_num_tiles, perf_total_num_tiles[0]);
   }
}

template <std::uint32_t wait_res=WaitRes::SRCA>
inline void math_record_input_start_time() {
   if constexpr(SKIP_UNP0) {
      TTI_SETDVALID(p_setrwc::SET_A);
   }
   if constexpr(SKIP_UNP1) {
      TTI_SETDVALID(p_setrwc::SET_B);
   }
   if (record_perf_events) {
      start_fpu_perf_cnt<true, wait_res>();
   }
}

inline void math_record_input_end_time() {
   if constexpr(SKIP_UNP0) {
      TTI_CLEARDVALID(0x1, 0);
   }
   if constexpr(SKIP_UNP1) {
      TTI_CLEARDVALID(0x2, 0);
   }
   if (record_perf_events) {
      stop_fpu_perf_cnt<true, true>();
      record_fpu_perf_cnt_value();
   }
}

template <std::uint32_t wait_res=WaitRes::SRCA>
inline void set_max_unpack_first_block_to_record() {
   if constexpr (wait_res == WaitRes::SRCA || wait_res == WaitRes::SRCB) {
      max_unpack_first_block_to_record = 1;
   } else {
      max_unpack_first_block_to_record = 2;
   }
   num_unpack_first_block_recorded = 0;
}

}

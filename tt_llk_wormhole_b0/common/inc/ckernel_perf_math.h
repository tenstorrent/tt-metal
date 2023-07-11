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

#ifndef INTERMED_DUMP
#define INTERMED_DUMP 0
#endif

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
extern uint16_t current_outer_loop_iter;
extern uint8_t thread_id;
extern uint32_t perf_events_target_idx;

// In math thread, THCON dumps perf buffers in l1.
// Therefore, incrementing the ncrisc perf_dram_buffer_req must be done by THCON as well.
// Flipping the l1 perf start address must also be done by THCON for math thread.
// Following variable keeps track of latest value of perf_dram_copy_req[1] from trisc perspective.
// The actual value might be different, because the queued THCON updates for perf_dram_copy_req[1] might have yet not been executed.
// We read this value initially for all threads to reduce the l1-reads.
extern int32_t dram_dump_req_local;

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

inline void set_perf_cnt_params(uint32_t block_sel=cperf_cnt_block_sel::PERF_CNT_FPU, uint32_t ref_period=0xffffffff, uint32_t mode=cperf_cnt_mode::PERF_CNT_MODE_FREE) {
  uint32_t perf_cnt_ref_period_reg;
  switch (block_sel) {
     case cperf_cnt_block_sel::PERF_CNT_INSTR_THREAD:     perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0; break;
     case cperf_cnt_block_sel::PERF_CNT_L1:              perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
     default: perf_cnt_ref_period_reg = RISCV_DEBUG_REG_PERF_CNT_FPU0;
  }
  reg_write(perf_cnt_ref_period_reg, ref_period);
  reg_write(perf_cnt_ref_period_reg+4, 0x00010100);
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
   if (perf_events_target_idx <= 1) {
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

template <bool use_tensix=true>
inline void start_fpu_perf_cnt() {
   if (perf_events_target_idx <= 1) {
      if constexpr (use_tensix) {
            TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
            TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::THCON);
            TTI_STOREREG(p_gpr_math::PERF_CNT_START, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
            TTI_STOREREG(p_gpr::ZERO, (RISCV_DEBUG_REG_PERF_CNT_FPU2>>2)&0x3ffff);
      } else {
         reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000001);
         reg_write(RISCV_DEBUG_REG_PERF_CNT_FPU2, 0x00000000);
      }
   }
}


inline void sel_fpu_perf_cnt(uint32_t cnt_id) {
   riscv_debug_reg_dbg_dbus_cntl_u dbg_bus_cntl;
   dbg_bus_cntl.val = reg_read(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);;
   dbg_bus_cntl.f.dbg_daisy_sel = cperf_dbg_daisy_id::DEBUG_DAISY_TENSIX;
   dbg_bus_cntl.f.dbg_sig_sel   = 0x0;
   dbg_bus_cntl.f.dbg_rd_sel    = cnt_id<<1; //rd_sel is aligned to 16-bit
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

// Return value of the selected perf counter
inline uint32_t get_perf_cnt() {
   return reg_read(RISCV_DEBUG_REG_DBG_RD_DATA);
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
   dbg_bus_cntl.val = reg_read(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);
   dbg_bus_cntl.f.dbg_reg_ovrd_en = 0x1;
   dbg_bus_cntl.f.dbg_daisy_en = 0x1;
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

inline void dbg_daisy_disable() {
   riscv_debug_reg_dbg_dbus_cntl_u dbg_bus_cntl;
   dbg_bus_cntl.val = reg_read(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG);
   dbg_bus_cntl.f.dbg_reg_ovrd_en = 0x0;
   dbg_bus_cntl.f.dbg_daisy_en = 0x0;
   reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
}

inline void dbg_enable_dump_to_mem(uint32_t start_addr, uint32_t end_addr) {

   TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
   uint32_t start_addr_lo = (start_addr >> 4) & 0xffff;
   uint32_t start_addr_hi = (start_addr >> 4) >> 16;
   TT_SETDMAREG(0, start_addr_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, start_addr_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG0 >> 2) & 0x3ffff);

   uint32_t end_addr_lo = (end_addr >> 4) & 0xffff;
   uint32_t end_addr_hi = (end_addr >> 4) >> 16;
   TT_SETDMAREG(0, end_addr_lo, 0, LO_16(p_gpr_math::TMP0));
   TT_SETDMAREG(0, end_addr_hi, 0, HI_16(p_gpr_math::TMP0));
   TTI_STOREREG(p_gpr_math::TMP0, (RISCV_DEBUG_REG_DBG_L1_MEM_REG1 >> 2) & 0x3ffff);

   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG0, start_addr>>4);
   // reg_write(RISCV_DEBUG_REG_DBG_L1_MEM_REG1, end_addr>>4);
   riscv_debug_reg_dbg_l1_mem_reg2_u dbg_l1_mem_reg2;
   dbg_l1_mem_reg2.val = 0;
   dbg_l1_mem_reg2.f.mem_dump_mode = 0xf; //invalid and overriden below to trigger pulse needed to latch start address
   dbg_l1_mem_reg2.f.skip_cycles = 0;

   uint32_t debug_l1_reg2_lo = dbg_l1_mem_reg2.val & 0xffff;
   uint32_t debug_l1_reg2_hi = (dbg_l1_mem_reg2.val >> 16) & 0xffff;
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

template <bool use_tensix=true>
inline void record_fpu_perf_cnt_value() {
   // if (perf_events_target_idx <= 1) {
   //    // In l1 mode always reserve last event for PERF_DUMP_END_SIGNAL.
   //    uint32_t reserve_space_for_trisc_end_signal = 1;
   //    if (perf_index + 3 <= perf_end-reserve_space_for_trisc_end_signal) { // Last event is always set to a default.
   //       //perf_buf_base[perf_index] = get_perf_cnt();
   //       //perf_buf_base[perf_index + 1] = get_perf_cnt();
   //       dump_perf_cnt_to_mem<use_tensix>(); //Dump 16B to L1
   //       perf_index += (PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t));
   //    }
   // }
}

// Dump a dummy math event to get the initial fpu counter value.
inline void record_dummy_math_event() {
   if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      uint32_t reserve_space_for_trisc_end_signal = 1;
      if (perf_index + 3 <= perf_end-reserve_space_for_trisc_end_signal) { // Last event is always set to a default.
         perf_buf_base[perf_buf_base_id][perf_index] = 0;
         perf_buf_base[perf_buf_base_id][perf_index+1] = 0;
         perf_buf_base[perf_buf_base_id][perf_index+2] = 0;
         perf_buf_base[perf_buf_base_id][perf_index+3] = 0;
         perf_index += (PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t));
      }
   }
}

inline void setup_fpu_perf_cnt() {
   // Only program perf counters for math thread (trisc1)
   if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      set_perf_cnt_params(cperf_cnt_block_sel::PERF_CNT_FPU,0xffffffff,cperf_cnt_mode::PERF_CNT_MODE_FREE);
      sel_fpu_perf_cnt(0);
      dbg_daisy_enable();
      dbg_enable_dump_to_mem((uint32_t)&perf_buf_base[0][PERF_CNT_DUMP_ENTRY_SIZE/sizeof(uint32_t)], (uint32_t)&perf_buf_base[0][perf_end]);

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
}

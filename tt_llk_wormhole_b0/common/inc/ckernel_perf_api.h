#pragma once

#include <cstdint>
#include <l1_address_map.h>
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "tensix.h"
#include "fw_debug.h"
#include "epoch.h"

#ifdef PERF_DUMP
#include "perf_events_target_inputs.h"
#include "perf_lib/scratch_api.h"
#include "perf_res_decouple.h"
#include "ckernel_perf_math.h"
#include "ckernel_perf_unpack_pack.h"
#endif

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
extern bool record_perf_events;
extern uint32_t perf_events_target_idx;
extern uint16_t current_outer_loop_iter;
extern uint8_t thread_id;
extern bool first_unpack_recorded;

inline void set_perf_dump_flag_for_input(int input_idx) {
   if (perf_events_target_inputs[perf_events_target_idx] == input_idx) {
      record_perf_events = true;
      perf_events_target_idx++;
      if constexpr (PERF_DUMP_CONCURRENT) {
         if (thread_id == 0 || thread_id == 2) {
            perf_end = TRISC_PERF_BUF_SIZE >> 3;
         }
      } else {
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
      }
      current_outer_loop_iter = input_idx;
   } else {
      record_perf_events = false;
   }
   first_unpack_recorded = false;
}

inline void record_pack_input_init_timestamp() {
   if (record_perf_events) {
      uint32_t event_id = perf::get_event_id(0, 0, perf::EventType::PACK_EACH_INPUT, current_outer_loop_iter);
      record_timestamp_64b(event_id);
   }
}

void record_pack_input_end_timestamp() {
   if (record_perf_events) {
      uint32_t event_id = perf::get_event_id(0, 0, perf::EventType::PACK_EACH_INPUT, current_outer_loop_iter);
      record_timestamp_64b(event_id);
      if (perf_events_target_idx == 1) {
         uint32_t event_id_num_tiles_pack = perf::get_event_id(0, 0, perf::EventType::NUM_TILES_PACK, 0);
         uint16_t num_tiles = regfile[p_gpr_pack::PERF_PACK_NUM_TILES] & 0xffff;
         record_perf_value_and_check_overflow(event_id_num_tiles_pack, num_tiles, 0);
      }
   }
}

inline void perf_math_counter_start() {
   if constexpr(SKIP_UNP) {
      TTI_SETDVALID(p_setrwc::SET_A);
      TTI_SETDVALID(p_setrwc::SET_B);
   }
   if (record_perf_events) {
      // Due to a race condition that corrupts the write address of the fpu counters, reprogram them for every input
      dbg_enable_dump_to_mem((uint32_t)&perf_buf_base[perf_buf_base_id][perf_index], (uint32_t)&perf_buf_base[perf_buf_base_id][perf_end]);
      start_fpu_perf_cnt<true>();
   }
}

inline void record_perf_math_counter() {
   if constexpr(SKIP_UNP) {
      TTI_CLEARDVALID(0x1, 0);
      TTI_CLEARDVALID(0x2, 0);
   }
   if (record_perf_events) {
      stop_fpu_perf_cnt<true, true>();
      // record_fpu_perf_cnt_value();
   }
}

void record_unpack_num_tiles() {
   if (perf_events_target_idx == 1) {
      for (uint8_t operand = 0; operand < PERF_MAX_NUM_INPUTS; operand++) {
         uint regfile_base_idx = p_gpr_unpack::PERF_UNPACK_NUM_TILES_0;
         regfile_base_idx += (operand >> 1);
         bool upper = operand & 0b1;
         uint16_t num_tiles;
         if (upper) {
            num_tiles = (regfile[regfile_base_idx] >> 16) & 0xffff;
         } else {
            num_tiles = regfile[regfile_base_idx] & 0xffff;
         }
         if (num_tiles != 0) {
            uint32_t event_id_num_tiles_unpack = perf::get_event_id(operand, 0, perf::EventType::NUM_TILES_UNPACK, 0);
            record_perf_value_and_check_overflow(event_id_num_tiles_unpack, num_tiles, 0);
         }
      }
   }
}

void record_unpack_first_instruction_timestamp() {
   if (record_perf_events) {
      uint32_t clock_lo = regfile[p_gpr_unpack::PERF_FIRST_UNP_LO];
      uint32_t clock_hi = regfile[p_gpr_unpack::PERF_FIRST_UNP_HI];
      uint32_t event_id_last_wait_tile = perf::get_event_id(0, 0, perf::EventType::UNPACK_FIRST_INSTRUCTION, current_outer_loop_iter);
      record_perf_value_and_check_overflow(event_id_last_wait_tile, clock_lo, clock_hi);
      if (perf_events_target_idx == 1) {
         record_unpack_num_tiles();
      }
   }
}

}

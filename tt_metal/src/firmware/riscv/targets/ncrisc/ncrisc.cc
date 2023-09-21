// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "run_sync.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"

#include "debug_status.h"

#include "debug_print.h"

volatile uint32_t local_mem_barrier __attribute__((used));

uint8_t mailbox_index = 0;
uint8_t mailbox_end = 32;

uint32_t halt_stack_ptr_save;

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t noc_size_x __attribute__((used));
uint8_t noc_size_y __attribute__((used));

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

extern "C" void ncrisc_resume(void);
extern "C" void notify_brisc_and_halt(uint32_t ncrisc_go_toggle);

int main(int argc, char *argv[]) {

  DEBUG_STATUS('I');

  int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
  l1_to_local_mem_copy((uint*)__ldm_data_start, (uint tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE, num_words);


  risc_init();
  *(volatile tt_l1_ptr uint32_t *)MEM_NCRISC_RESUME_ADDR_MAILBOX_ADDRESS = (uint32_t)ncrisc_resume;

  // Cleanup profiler buffer incase we never get the go message
  kernel_profiler::init_profiler();
  while (1) {

      DEBUG_STATUS('W');
      notify_brisc_and_halt(RUN_SYNC_MESSAGE_DONE);

      kernel_profiler::init_profiler();
      kernel_profiler::mark_time(CC_MAIN_START);

      DEBUG_STATUS('R');
      kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
      kernel_init();
      kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
      DEBUG_STATUS('D');

      kernel_profiler::mark_time(CC_MAIN_END);
  }

  return 0;
}

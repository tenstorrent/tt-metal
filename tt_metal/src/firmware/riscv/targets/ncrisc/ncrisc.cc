#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"


volatile uint32_t local_mem_barrier __attribute__((used));
volatile tt_l1_ptr uint32_t* const run_mailbox_address = (volatile tt_l1_ptr uint32_t*)(MEM_RUN_MAILBOX_ADDRESS + MEM_MAILBOX_NCRISC_OFFSET);

volatile tt_l1_ptr uint16_t *debug_mailbox_base = nullptr;
uint8_t mailbox_index = 0;
uint8_t mailbox_end = 32;

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t noc_size_x __attribute__((used));
uint8_t noc_size_y __attribute__((used));

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

inline void record_mailbox_value(uint16_t event_value) {
  if (mailbox_index < mailbox_end) {
    debug_mailbox_base[mailbox_index] = event_value;
    mailbox_index++;
  }
}

inline void record_mailbox_value_with_index(uint8_t index, uint16_t event_value) {
  if (index < mailbox_end) {
    debug_mailbox_base[index] = event_value;
  }
}

inline void record_mailbox_value_l1(uint16_t event_value) {
  if (mailbox_index < mailbox_end) {
    debug_mailbox_base[mailbox_index] = event_value;
    mailbox_index++;
  }
}

inline void record_mailbox_value_with_index_l1(uint8_t index, uint16_t event_value) {
  if (index < mailbox_end) {
    debug_mailbox_base[index] = event_value;
  }
}

inline void allocate_debug_mailbox_buffer() {
  std::int32_t debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + 3*MEM_DEBUG_MAILBOX_SIZE;
  debug_mailbox_base = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(debug_mailbox_addr);
}

int main(int argc, char *argv[]) {
  int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
  l1_to_local_mem_copy((uint*)__ldm_data_start, (uint*)MEM_NCRISC_INIT_LOCAL_L1_BASE, num_words);

  kernel_profiler::init_profiler();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_START);
#endif
  allocate_debug_mailbox_buffer();

  risc_init();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
#endif
  kernel_init();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif

  *run_mailbox_address = 0x1;

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_END);
#endif
  while (true);
  return 0;
}

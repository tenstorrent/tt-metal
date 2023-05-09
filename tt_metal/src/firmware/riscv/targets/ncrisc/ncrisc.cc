#include "context.h"
#include "risc_common.h"
#include "l1_address_map.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"

volatile uint32_t local_mem_barrier;
volatile uint32_t* test_mailbox_ptr = (volatile uint32_t*)(MEM_NCRISC_FIRMWARE_BASE + MEM_TEST_MAILBOX_ADDRESS);

int post_index;

volatile uint32_t noc_read_scratch_buf[32] __attribute__((section("data_l1_noinit"))) __attribute__((aligned(32))) ;
volatile uint16_t *debug_mailbox_base = nullptr;
uint8_t mailbox_index = 0;
uint8_t mailbox_end = 32;

uint8_t my_x[NUM_NOCS];
uint8_t my_y[NUM_NOCS];
#ifdef NOC_INDEX
uint8_t loading_noc = NOC_INDEX;
#else
uint8_t loading_noc = 1; // NCRISC uses NOC-1
#endif
uint8_t noc_size_x;
uint8_t noc_size_y;

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

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

inline __attribute__((section("code_l1"))) void record_mailbox_value_l1(uint16_t event_value) {
  if (mailbox_index < mailbox_end) {
    debug_mailbox_base[mailbox_index] = event_value;
    mailbox_index++;
  }
}

inline __attribute__((section("code_l1"))) void record_mailbox_value_with_index_l1(uint8_t index, uint16_t event_value) {
  if (index < mailbox_end) {
    debug_mailbox_base[index] = event_value;
  }
}

inline void allocate_debug_mailbox_buffer() {
  std::int32_t debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + 3*MEM_DEBUG_MAILBOX_SIZE;
  debug_mailbox_base = reinterpret_cast<volatile uint16_t *>(debug_mailbox_addr);
}

void local_mem_copy() {
   volatile uint *l1_local_mem_start_addr;
   volatile uint *local_mem_start_addr = (volatile uint*) MEM_LOCAL_BASE;

   if ((uint)__firmware_start == (uint)MEM_NCRISC_FIRMWARE_BASE) {
      l1_local_mem_start_addr = (volatile uint*)l1_mem::address_map::NCRISC_LOCAL_MEM_BASE;
   }
   uint word_num = ((uint)__local_mem_rodata_end_addr - (uint)__local_mem_rodata_start_addr)>>2;

   if (word_num>0) {
      for (uint n=0;n<word_num;n++) {
         local_mem_start_addr[n] = l1_local_mem_start_addr[n];
      }
      local_mem_barrier = l1_local_mem_start_addr[word_num-1]; // TODO - share via ckernel.h?
   }
}

#include "dataflow_api.h"
#include "kernel.cpp"

int main(int argc, char *argv[]) {
  kernel_profiler::init_profiler();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_START);
#endif
  init_riscv_context();
  allocate_debug_mailbox_buffer();

  if ((uint)MEM_LOCAL_BASE == ((uint)__local_mem_rodata_end_addr&0xfff00000))
  {
      local_mem_copy();
  }

  noc_init(loading_noc); // NCRISC uses NOC-1
  risc_init();

  setup_cb_read_write_interfaces();

#if defined(IS_DISPATCH_KERNEL)
    setup_cq_read_write_interface();
#endif

  init_dram_bank_to_noc_coord_lookup_tables();
  init_l1_bank_to_noc_coord_lookup_tables();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
#endif
  kernel_main();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif

  test_mailbox_ptr[0] = 0x1;

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_END);
#endif
  while (true);
  return 0;
}

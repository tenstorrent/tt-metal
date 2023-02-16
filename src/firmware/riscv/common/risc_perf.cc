#include "epoch.h"
#include "l1_address_map.h"
#include "tensix.h"
#include "noc_nonblocking_api.h"
#include "risc_perf.h"
#include "risc_common.h"

namespace risc
{
uint32_t perf_dram_initialized = 0;
uint32_t perf_index __attribute__((section(".bss"))) = 0;
uint32_t perf_end __attribute__((section(".bss"))) = 0;
uint32_t wall_clk_h __attribute__((section(".bss"))) = 0;
volatile uint32_t *perf_double_buf_base[2] = {nullptr, nullptr};
volatile uint32_t *perf_buf_base;
uint64_t thread_dram_ptr[l1_mem::address_map::PERF_NUM_THREADS];
uint16_t thread_req_max[l1_mem::address_map::PERF_NUM_THREADS];
uint32_t thread_dram_copy_ack[l1_mem::address_map::PERF_NUM_THREADS];
uint32_t thread_l1_addr_l[l1_mem::address_map::PERF_NUM_THREADS];
uint32_t thread_l1_addr_h[l1_mem::address_map::PERF_NUM_THREADS];
uint16_t thread_l1_buf_sel[l1_mem::address_map::PERF_NUM_THREADS];
volatile uint32_t epoch_perf_scratch[PERF_START_OFFSET] __attribute__((section("data_l1_noinit"))) __attribute__((aligned(32))) ;


void init_perf_dram_state() {
  if (perf_dram_initialized == 0) {
    for (int i = 0; i < l1_mem::address_map::PERF_NUM_THREADS; i++) {
      thread_dram_ptr[i] = EPOCH_INFO_PTR->perf_dram_addr[i];
      thread_req_max[i] = EPOCH_INFO_PTR->perf_req_max[i]; 
      thread_dram_copy_ack[i] = 0;

      //Reset thread l1 half buffer select.
	
      thread_l1_buf_sel[i] = 0;
    }
    thread_l1_addr_l[0] = T0_PERF_L;
    thread_l1_addr_l[1] = T1_PERF_L;
    thread_l1_addr_l[2] = T2_PERF_L;
    thread_l1_addr_l[3] = T3_PERF_L;
    thread_l1_addr_h[0] = T0_PERF_H;
    thread_l1_addr_h[1] = T1_PERF_H;
    thread_l1_addr_h[2] = T2_PERF_H;
    thread_l1_addr_h[3] = T3_PERF_H;

    perf_dram_initialized = 1;
  } else {
    for (int i = 0; i < l1_mem::address_map::PERF_NUM_THREADS; i++) {
      //When epoch starts, restore the req/ack conters as they do not reset across epochs.
      EPOCH_INFO_PTR->perf_dram_copy_req[i] = thread_dram_copy_ack[i];
      EPOCH_INFO_PTR->perf_dram_copy_ack[i] = thread_dram_copy_ack[i];
            //Reset thread l1 half buffer select.
      thread_l1_buf_sel[i] = 0;

    }
    //For risc, update new epoch's dram buf start address.
    //This address is used by risc at the end of epoch to spill epoch perf scratch buffer.
    //Set perf_dram_addr to 0 in dram buffer is already full from previous epochs.
    //NCRISC will use perf_dram_addr == 0 as a flag to not write epoch scratch buffer.
    if (thread_dram_copy_ack[3] < thread_req_max[3]) {
      EPOCH_INFO_PTR->perf_dram_addr[3] = thread_dram_ptr[3];
    } else {
      EPOCH_INFO_PTR->perf_dram_addr[3] = 0;
    }
  }
}

void allocate_perf_buffer() {
  perf_index = PERF_START_OFFSET; // The first 4B value is always initialized to 0xbaddf00d. Also account for 6 fixed epoch related events.
  perf_double_buf_base[0] = reinterpret_cast<volatile uint32_t *>(l1_mem::address_map::NCRISC_L1_PERF_BUF_BASE); // Start address of lower half buffer.
  perf_double_buf_base[1] = reinterpret_cast<volatile uint32_t *>(l1_mem::address_map::NCRISC_L1_PERF_BUF_BASE + (NCRISC_PERF_BUF_SIZE >> 1)); //Start address of upper half buffer
  perf_buf_base = perf_double_buf_base[0]; // Start dumping from lower half of L1 perf buffer.
  if constexpr (INTERMED_DUMP) {
    epoch_perf_scratch[0] = PERF_DUMP_END_SIGNAL;
    perf_end = NCRISC_PERF_BUF_SIZE >> 3; // 4 bytes per event for half buffer
    if constexpr (PERF_DUMP_LEVEL == 0) {
      perf_buf_base[0] = PERF_DUMP_END_SIGNAL;
      perf_end = NCRISC_PERF_BUF_SIZE >> 2; // 4 bytes per event, no double buffer in this mode.
    }

  } else {
    perf_buf_base[0] = PERF_DUMP_END_SIGNAL;
    perf_end = NCRISC_PERF_BUF_SIZE >> 2; // 4 bytes per event
  }
  if constexpr (PERF_DUMP_LEVEL != 0) {
    perf_buf_base[perf_end-1] = PERF_DUMP_PADDING;
    perf_buf_base[perf_end-2] = PERF_DUMP_PADDING;
  }
  uint32_t temp = reg_read_barrier_l1(RISCV_DEBUG_REG_WALL_CLOCK_L);
  wall_clk_h = reg_read_barrier_l1(RISCV_DEBUG_REG_WALL_CLOCK_H);
  record_perf_value_l1(perf_event::WALL_CLOCK_TOP_32B, wall_clk_h);
  
}

void __attribute__ ((noinline)) record_perf_dump_end() {
  if (perf_index < perf_end) {
    perf_buf_base[perf_index] = PERF_DUMP_END_SIGNAL;
    perf_index += 1;
  }
  if constexpr (INTERMED_DUMP) {
    EPOCH_INFO_PTR->perf_dram_copy_req[3]++;
  } else {
    EPOCH_INFO_PTR->perf_dram_copy_req[3] += 2;
  }
  check_dram_spill_requests();
  if constexpr (PERF_DUMP_LEVEL != 0 && INTERMED_DUMP) {
    spill_risc_epoch_perf_scratch();
  }
  
}

void record_timestamp(uint32_t event_id) {
  uint32_t low;
  uint32_t high;
  low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
  high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
  if (high != wall_clk_h) {
    wall_clk_h = high;
    //record_perf_value(perf_event::WALL_CLOCK_TOP_32B, high);
    record_perf_value_and_check_overflow(perf_event::WALL_CLOCK_TOP_32B, high);
  }
  //record_perf_value(event_id, low);
  record_perf_value_and_check_overflow(event_id, low);
}

void record_timestamp_at_offset(uint32_t event_id, uint32_t offset) {
  
  uint32_t low;
  uint32_t high;
  low = reg_read_barrier_l1(RISCV_DEBUG_REG_WALL_CLOCK_L);
  high = reg_read_barrier_l1(RISCV_DEBUG_REG_WALL_CLOCK_H);
  record_perf_value_at_offset(event_id, low, high, offset);
  
}

void record_timestamp_at_offset_l1(uint32_t event_id, uint32_t offset) {
  
  uint32_t low;
  uint32_t high;
  low = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
  high = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
  record_perf_value_at_offset(event_id, low, high, offset);
  
}

void check_dram_spill_requests() {
  for (int i = 0; i < l1_mem::address_map::PERF_NUM_THREADS; i++) {
    if (EPOCH_INFO_PTR->perf_dram_copy_req[i] != thread_dram_copy_ack[i]) {
      if (thread_dram_copy_ack[i] < thread_req_max[i]) {
        uint32_t l1_address = thread_l1_buf_sel[i] ? thread_l1_addr_h[i] : thread_l1_addr_l[i];
        if constexpr (INTERMED_DUMP) {
          ncrisc_noc_fast_write_any_len_l1(PERF_DUMP_NOC, NCRISC_WR_CMD_BUF, l1_address, thread_dram_ptr[i], TRISC_PERF_BUF_SIZE/2, PERF_DUMP_VC, false, false, 1);
          thread_dram_ptr[i] += TRISC_PERF_BUF_SIZE/2;
          thread_l1_buf_sel[i] = 1 - thread_l1_buf_sel[i]; //toggle thread_l1_buf_sel for next time.
        } else {
          ncrisc_noc_fast_write_any_len_l1(PERF_DUMP_NOC, NCRISC_WR_CMD_BUF, l1_address, thread_dram_ptr[i], TRISC_PERF_BUF_SIZE, PERF_DUMP_VC, false, false, 1);
          thread_dram_ptr[i] += TRISC_PERF_BUF_SIZE;
        }
      }
      if constexpr (INTERMED_DUMP) {
        thread_dram_copy_ack[i]++;
      } else {
        thread_dram_copy_ack[i] += 2;
      }
    }
  }
  while (!ncrisc_noc_nonposted_writes_sent_l1(PERF_DUMP_NOC)){};
  for (int i = 0; i < l1_mem::address_map::PERF_NUM_THREADS; i++) {
    EPOCH_INFO_PTR->perf_dram_copy_ack[i] = thread_dram_copy_ack[i];
  }
}

}


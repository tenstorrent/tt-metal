#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_kernel_api.h"
#include "tensix_functions.h"

#include "kernel.cpp"

CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];
CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
CQReadInterface cq_read_interface;

#ifdef NOC_INDEX
uint8_t loading_noc = NOC_INDEX;
#else
uint8_t loading_noc = 1;
#endif

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

// dram channel to x/y lookup tables
// The number of banks is generated based off device we are running on --> controlled by allocator
uint8_t dram_bank_to_noc_x[NUM_DRAM_BANKS];
uint8_t dram_bank_to_noc_y[NUM_DRAM_BANKS];
uint32_t dram_bank_to_noc_xy[NUM_DRAM_BANKS];

uint8_t l1_bank_to_noc_x[NUM_L1_BANKS];
uint8_t l1_bank_to_noc_y[NUM_L1_BANKS];
uint32_t l1_bank_to_noc_xy[NUM_L1_BANKS];

void kernel_launch() {

    firmware_kernel_common_init((void *)MEM_NCRISC_INIT_LOCAL_L1_BASE);

    dataflow_internal::setup_cb_read_write_interfaces();

    dataflow_internal::init_dram_bank_to_noc_coord_lookup_tables();
    dataflow_internal::init_l1_bank_to_noc_coord_lookup_tables();

    noc_init(loading_noc);

    kernel_main();
}

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_kernel_api.h"
#include "debug_print.h"

#include <kernel.cpp>

#include "debug_print.h"

CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];
CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
CQReadInterface cq_read_interface;

#ifdef NOC_INDEX
uint8_t loading_noc = NOC_INDEX;
#else
uint8_t loading_noc = 0;
#endif

// to reduce the amount of code changes, both BRISC and NRISCS instatiate these counter for both NOCs (ie NUM_NOCS)
// however, atm NCRISC uses only NOC-1 and BRISC uses only NOC-0
// this way we achieve full separation of counters / cmd_buffers etc.
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

extern uint64_t dispatch_addr;
extern uint8_t kernel_noc_id_var;

inline void notify_host_kernel_finished() {
    uint32_t pcie_noc_x = NOC_X(0);
    uint32_t pcie_noc_y = NOC_Y(4); // These are the PCIE core coordinates
    uint64_t pcie_address =
        dataflow::get_noc_addr(pcie_noc_x, pcie_noc_y, 0);  // For now, we are writing to host hugepages at offset 0 (nothing else currently writing to it)

    volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(NOTIFY_HOST_KERNEL_COMPLETE_ADDR);
    done[0] = NOTIFY_HOST_KERNEL_COMPLETE_VALUE; // 512 was chosen arbitrarily, but it's less common than 1 so easier to check validity

    // Write to host hugepages to notify of completion
    dataflow::noc_async_write(NOTIFY_HOST_KERNEL_COMPLETE_ADDR, pcie_address, 4);
    dataflow::noc_async_write_barrier();
}

void kernel_launch() {

    firmware_kernel_common_init((void *)MEM_BRISC_INIT_LOCAL_L1_BASE);

#if defined(IS_DISPATCH_KERNEL)
    dataflow_internal::setup_cq_read_write_interface();
#else
    dataflow_internal::setup_cb_read_write_interfaces();                // done by both BRISC / NCRISC
#endif

    dataflow_internal::init_dram_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC
    dataflow_internal::init_l1_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC

    noc_init(loading_noc);

    kernel_main();

    // FW needs to notify device dispatcher when we are done
    // Some information needed is known here, pass it back
    kernel_noc_id_var = loading_noc;
#if defined(TT_METAL_DEVICE_DISPATCH_MODE)
    dispatch_addr = (my_x[loading_noc] == NOC_X(1) && my_y[loading_noc] == NOC_Y(11)) ?
        0 :
        dataflow::get_noc_addr(1, 11, DISPATCH_MESSAGE_ADDR);
#else
    dispatch_addr = 0;
#endif
}

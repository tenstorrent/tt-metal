// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "noc_addr_ranges_gen.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"
#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"

#include "debug/status.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"

uint8_t noc_index;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t atomic_ret_val __attribute__ ((section ("l1_data"))) __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

//c_tensix_core core;

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);

constexpr uint32_t num_cbs_to_early_init = 4;  // safe small number to overlap w/ ncrisc copy

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
    uint16_t core_flat_id __attribute__((used));
}
#endif

//inline void RISC_POST_STATUS(uint32_t status) {
//  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
//  ptr[0] = status;
//}

void init_sync_registers() {
    volatile tt_reg_ptr uint* tiles_received_ptr;
    volatile tt_reg_ptr uint* tiles_acked_ptr;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
      tiles_received_ptr = get_cb_tiles_received_ptr(operand);
      tiles_received_ptr[0] = 0;
      tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
      tiles_acked_ptr[0] = 0;
    }
}

void flush_icache() {
#ifdef ARCH_BLACKHOLE
    // Kernel start instructions on WH are not cached because we apply a 1 cache line (32B) padding
    //  between FW end and Kernel start.
    // This works because risc tries to prefetch 1 cache line.
    // The 32B still get cached but they are never executed
    #pragma GCC unroll 2048
    for (int i = 0; i < 2048; i++) {
        asm("nop");
    }
#endif
}

int main() {
    disable_lowcache();
    DIRTY_STACK_MEMORY();
    DEBUG_STATUS("I");
    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    uint32_t *local_mem_ptr = (uint32_t *)__ldm_data_start;
    uint32_t *l1_data_ptr = (uint32_t *)MEM_IERISC_INIT_LOCAL_L1_BASE;
    uint32_t heartbeat = 0;
    for (int32_t i = 0; i < num_words; i++) {
        local_mem_ptr[i] = l1_data_ptr[i];
    }

    risc_init();
    //device_setup();
    noc_init();

    mailboxes->launch.go.run = RUN_MSG_DONE;

    // Cleanup profiler buffer incase we never get the go message
    while (1) {

        init_sync_registers();
        // Wait...
        DEBUG_STATUS("GW");
        while (mailboxes->launch.go.run != RUN_MSG_GO)
        {
            RISC_POST_HEARTBEAT(heartbeat);
        };
        DEBUG_STATUS("GD");

        {
            DeviceZoneScopedMainN("ERISC-IDLE-FW");
            DeviceZoneSetCounter(mailboxes->launch.kernel_config.host_assigned_id);

            noc_index = mailboxes->launch.kernel_config.brisc_noc_id;

            //UC FIXME: do i need this?
            setup_cb_read_write_interfaces(0, num_cbs_to_early_init, true, true);

            // Run the ERISC kernel
            DEBUG_STATUS("R");
            //if (mailboxes->launch.enable_brisc) {
                //UC FIXME: do i need this?
                setup_cb_read_write_interfaces(num_cbs_to_early_init, mailboxes->launch.kernel_config.max_cb_index, true, true);
                uint32_t kernel_config_base = mailboxes->launch.kernel_config.kernel_config_base;
                rta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
                    mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_ETH_DM0].rta_offset);
                crta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
                    mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_ETH_DM0].crta_offset);

                flush_icache();
                kernel_init();
                RECORD_STACK_USAGE();
            //} else {
                // This was not initialized in kernel_init
            //    noc_local_state_init(noc_index);
            //}
            DEBUG_STATUS("D");

            mailboxes->launch.go.run = RUN_MSG_DONE;


            // Notify dispatcher core that it has completed
            if (mailboxes->launch.kernel_config.mode == DISPATCH_MODE_DEV) {
                uint64_t dispatch_addr =
                    NOC_XY_ADDR(NOC_X(mailboxes->launch.kernel_config.dispatch_core_x),
                        NOC_Y(mailboxes->launch.kernel_config.dispatch_core_y), DISPATCH_MESSAGE_ADDR);
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
            }

            while (1) {
                RISC_POST_HEARTBEAT(heartbeat);
            }
        }
    }

    return 0;
}

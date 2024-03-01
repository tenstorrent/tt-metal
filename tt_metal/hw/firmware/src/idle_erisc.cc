// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

#include "debug/status.h"
#include "debug/dprint.h"

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

//c_tensix_core core;

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);

constexpr uint32_t num_cbs_to_early_init = 4;  // safe small number to overlap w/ ncrisc copy

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t device_function_sums[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
uint64_t device_function_starts[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
}

inline void RISC_POST_STATUS(uint32_t status) {
  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
  ptr[0] = status;
}

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

int main() {

    DEBUG_STATUS('I');
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

    mailboxes->launch.run = RUN_MSG_DONE;

    // Cleanup profiler buffer incase we never get the go message
    kernel_profiler::init_profiler();
    while (1) {

        init_sync_registers();
        // Wait...
        DEBUG_STATUS('G', 'W');
        while (mailboxes->launch.run != RUN_MSG_GO)
        {
            RISC_POST_HEARTBEAT(heartbeat);
        };
        DEBUG_STATUS('G', 'D');

        kernel_profiler::init_profiler();
        kernel_profiler::mark_time(CC_MAIN_START);

        uint32_t noc_index = mailboxes->launch.brisc_noc_id;

        //UC FIXME: do i need this?
        setup_cb_read_write_interfaces(0, num_cbs_to_early_init, true, true);

        // Run the ERISC kernel
        DEBUG_STATUS('R');
        //if (mailboxes->launch.enable_brisc) {
            //UC FIXME: do i need this?
            setup_cb_read_write_interfaces(num_cbs_to_early_init, mailboxes->launch.max_cb_index, true, true);
            kernel_init();
        //} else {
            // This was not initialized in kernel_init
        //    noc_local_state_init(noc_index);
        //}
        DEBUG_STATUS('D');

        mailboxes->launch.run = RUN_MSG_DONE;

        // Not including any dispatch related code
        kernel_profiler::mark_time(CC_MAIN_END);

        // Notify dispatcher core that it has completed
        if (mailboxes->launch.mode == DISPATCH_MODE_DEV) {
            uint64_t dispatch_addr = NOC_XY_ADDR(NOC_X(DISPATCH_CORE_X), NOC_Y(DISPATCH_CORE_Y), DISPATCH_MESSAGE_ADDR);
            noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
        }
    }

    return 0;
}

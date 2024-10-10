// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "dev_msgs.h"
#include "stream_io_map.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"

#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t *const slave_idle_erisc_run = &mailboxes->slave_sync.ncrisc;

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

inline __attribute__((always_inline)) void signal_slave_idle_erisc_completion() {
    *slave_idle_erisc_run = RUN_SYNC_MSG_DONE;
}

int main(int argc, char *argv[]) {
    conditionally_disable_l1_cache();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint *)__ldm_data_start, (uint tt_l1_ptr *)MEM_SLAVE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH, num_words);

    risc_init();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        DeviceZoneScopedMainN("SLAVE-IDLE-ERISC-FW");

        uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, DISPATCH_CLASS_ETH_DM1);
        uint32_t tt_l1_ptr *cb_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.cb_offset);
        setup_cb_read_write_interfaces(cb_l1_base, 0, mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.max_cb_index, true, true, false);

        WAYPOINT("R");
        kernel_init();
        RECORD_STACK_USAGE();
        WAYPOINT("D");

        signal_slave_idle_erisc_completion();
    }

    return 0;
}

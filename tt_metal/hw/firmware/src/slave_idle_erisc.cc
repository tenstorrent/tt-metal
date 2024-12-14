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
#include "circular_buffer.h"

#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t *const slave_idle_erisc_run = &mailboxes->slave_sync.dm1;

uint8_t noc_index = 0;  // TODO: hardcoding needed for profiler
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
    configure_l1_data_cache();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");
    do_crt1((uint32_t *)MEM_SLAVE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH);

    risc_init();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        while (*slave_idle_erisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        DeviceZoneScopedMainN("SLAVE-IDLE-ERISC-FW");

        flush_erisc_icache();

        uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, DISPATCH_CLASS_ETH_DM1);

        WAYPOINT("R");
        int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM1);
        void (*kernel_address)(uint32_t) = (void (*)(uint32_t))
            (kernel_config_base + mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index]);
        (*kernel_address)((uint32_t)kernel_address);
        RECORD_STACK_USAGE();
        WAYPOINT("D");

        signal_slave_idle_erisc_completion();
    }

    return 0;
}

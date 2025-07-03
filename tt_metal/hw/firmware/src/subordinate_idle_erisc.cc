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
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "circular_buffer.h"

#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t *const subordinate_idle_erisc_run = &mailboxes->subordinate_sync.dm1;

uint8_t noc_index = 0;  // TODO: hardcoding needed for profiler

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

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

inline __attribute__((always_inline)) void signal_subordinate_idle_erisc_completion() {
    *subordinate_idle_erisc_run = RUN_SYNC_MSG_DONE;
}

int main(int argc, char *argv[]) {
    configure_csr();
    WAYPOINT("I");
    do_crt1((uint32_t *)MEM_SUBORDINATE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_IERISC_BANK_TO_NOC_SCRATCH);

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    risc_init();
    signal_subordinate_idle_erisc_completion();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        while (*subordinate_idle_erisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        DeviceZoneScopedMainN("SUBORDINATE-IDLE-ERISC-FW");

        flush_erisc_icache();

        uint32_t kernel_config_base =
            firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, DISPATCH_CLASS_ETH_DM1);
        my_relative_x_ =
            my_logical_x_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_x;
        my_relative_y_ =
            my_logical_y_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_y;

        WAYPOINT("R");
        int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM1);
        uint32_t kernel_lma =
            kernel_config_base + mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index];
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        record_stack_usage(stack_free);
        WAYPOINT("D");

        signal_subordinate_idle_erisc_completion();
    }

    return 0;
}

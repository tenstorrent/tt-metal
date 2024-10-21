// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/watcher_common.h"

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

void __attribute__((section("erisc_l1_code.1"), noinline)) Application(void) {
    WAYPOINT("I");
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    // Not using firmware_kernel_common_init since it is copying to registers
    // TODO: need to find free space that routing FW is not using
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    risc_init();
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    WAYPOINT("REW");
    uint32_t count = 0;
    while (routing_info->routing_enabled != 1) {
        volatile uint32_t *ptr = (volatile uint32_t *)0xffb2010c;
        count++;
        *ptr = 0xAABB0000 | (count & 0xFFFF);
        internal_::risc_context_switch();
    }
    WAYPOINT("RED");

    mailboxes->launch_msg_rd_ptr = 0; // Initialize the rdptr to 0
    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        uint8_t go_message_signal = mailboxes->go_message.signal;
        if (go_message_signal == RUN_MSG_GO) {
            // Only include this iteration in the device profile if the launch message is valid. This is because all workers get a go signal regardless of whether
            // they're running a kernel or not. We don't want to profile "invalid" iterations.
            DeviceZoneScopedMainN("ERISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
            enum dispatch_core_processor_masks enables = (enum dispatch_core_processor_masks)launch_msg_address->kernel_config.enables;
            if (enables & DISPATCH_CLASS_MASK_ETH_DM0) {
                firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, DISPATCH_CLASS_ETH_DM0);
                kernel_init();
            }
            mailboxes->go_message.signal = RUN_MSG_DONE;

            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr =
                    NOC_XY_ADDR(NOC_X(mailboxes->go_message.master_x),
                                NOC_Y(mailboxes->go_message.master_y), DISPATCH_MESSAGE_ADDR);
                internal_::notify_dispatch_core_done(dispatch_addr);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
                // Only executed if watcher is enabled. Ensures that we don't report stale data due to invalid launch messages in the ring buffer
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
            }
            WAYPOINT("R");

        } else if (go_message_signal == RUN_MSG_RESET_READ_PTR) {
            // Reset the launch message buffer read ptr
            mailboxes->launch_msg_rd_ptr = 0;
            int64_t dispatch_addr =
                NOC_XY_ADDR(NOC_X(mailboxes->go_message.master_x),
                            NOC_Y(mailboxes->go_message.master_y), DISPATCH_MESSAGE_ADDR);
            mailboxes->go_message.signal = RUN_MSG_DONE;
            internal_::notify_dispatch_core_done(dispatch_addr);
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}

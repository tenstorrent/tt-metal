// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "debug/watcher_common.h"

extern "C" void ApplicationHandler(void);

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
    uint16_t core_flat_id __attribute__((used));
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
uint32_t atomic_ret_val __attribute__ ((section ("l1_data"))) __attribute__((used));

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
    noc_init();
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

// This is a bespoke setjmp/longjmp implementation. We do not use
// regular setjmp/longjmp as that uses a 304 byte buffer. We only need
// enough to save the callee-save registers (13). Making this function
// naked allows us to place the jmp buffer at SP, which means we do
// not need to record a separate offset between sp and the jmp buffer.
// The function relies on optimization to avoid unexpected register
// usage.

__attribute__((section("erisc_l1_code.0"), naked, optimize("Os")))
void ApplicationHandler(void) {

  { // ApplicationHander pseudo-scope
    // Save callee saves.
    __asm__ volatile("addi sp, sp, -13 * 4\n\t"
		     "sw ra, 0 * 4(sp)\n\t"
		     "sw s0, 1 * 4(sp)\n\t"
		     "sw s1, 2 * 4(sp)\n\t"
		     "sw s2, 3 * 4(sp)\n\t"
		     "sw s3, 4 * 4(sp)\n\t"
		     "sw s4, 5 * 4(sp)\n\t"
		     "sw s5, 6 * 4(sp)\n\t"
		     "sw s6, 7 * 4(sp)\n\t"
		     "sw s7, 8 * 4(sp)\n\t"
		     "sw s8, 9 * 4(sp)\n\t"
		     "sw s9, 10 * 4(sp)\n\t"
		     "sw s10, 11 * 4(sp)\n\t"
		     "sw s11, 12 * 4(sp)\n\t"
		     ::: "memory");

    // Load the mailbox slot into a callee-save register (we'll use it twice).
    register uint32_t slot asm("s0") = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE;

    // Record sp in the save slot.
    __asm__ volatile("sw sp, 0(%[save_slot])\n\t"
		     : : [save_slot] "r"(slot) : "memory");

    Application();

    // Load up the parameter value
    __asm__ volatile ("mv a0,%[save_slot]" :: [save_slot] "r"(slot));
  }

  // Place the erisc_early_exit inside this function! avoids a tail call.
  // void erisc_early_exit(uint32_t save_slot)
  __asm__ volatile(".global erisc_early_exit\n\t"
	  ".type erisc_early_exit,@function\n\t"
	  "erisc_early_exit:\n\t");

  { // erisc_early_exit pseudo-scope

    // Restore sp from the save slot.
    __asm__ volatile("lw sp, 0(a0)\n\t");

    // Restore callee saves.
    __asm__ volatile("lw ra, 0 * 4(sp)\n\t"
		     "lw s0, 1 * 4(sp)\n\t"
		     "lw s1, 2 * 4(sp)\n\t"
		     "lw s2, 3 * 4(sp)\n\t"
		     "lw s3, 4 * 4(sp)\n\t"
		     "lw s4, 5 * 4(sp)\n\t"
		     "lw s5, 6 * 4(sp)\n\t"
		     "lw s6, 7 * 4(sp)\n\t"
		     "lw s7, 8 * 4(sp)\n\t"
		     "lw s8, 9 * 4(sp)\n\t"
		     "lw s9, 10 * 4(sp)\n\t"
		     "lw s10, 11 * 4(sp)\n\t"
		     "lw s11, 12 * 4(sp)\n\t"
		     "addi sp, sp, 4 * 13\n\t");

    // And we're done
    __asm__ volatile("ret");
  }

  // Finish up erisc_early_exit.
  __asm__ volatile(".size erisc_early_exit,. - erisc_early_exit\n\t");
}

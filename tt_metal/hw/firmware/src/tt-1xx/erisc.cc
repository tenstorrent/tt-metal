// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/tunneling.h"
#include "internal/firmware_common.h"
#include "noc_parameters.h"
#include "internal/risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/watcher_common.h"

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
uint32_t traceCount __attribute__((used));
}  // namespace kernel_profiler
#endif

uint8_t noc_index = 0;  // TODO: remove hardcoding

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

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

#if defined(ARCH_WORMHOLE) && defined(ENABLE_IRAM)
void l1_to_erisc_iram_copy(volatile uint32_t* iram_load_reg) {
    // Trigger copy of code from L1 to IRAM.
    *iram_load_reg = eth_l1_mem::address_map::KERNEL_BASE >> 4;
    RISC_POST_STATUS(0x10000000);
}

void l1_to_erisc_iram_copy_wait(volatile uint32_t* iram_load_reg) {
    // Wait for copy to complete.
    while (*iram_load_reg & 0x1);
}

void iram_setup() {
    // Copy code from L1 to IRAM.
    volatile uint32_t* iram_load_reg = (volatile uint32_t*)(ETH_CTRL_REGS_START + ETH_CORE_IRAM_LOAD);

    *(volatile uint32_t*)0xFFBA0000 = 0x30000000;
    *(volatile uint32_t*)0xFFBA0004 = 0x6;

    l1_to_erisc_iram_copy(iram_load_reg);
    l1_to_erisc_iram_copy_wait(iram_load_reg);

    *(volatile uint32_t*)0xFFBA0000 = 0x30000001;
    *(volatile uint32_t*)0xFFBA0004 = 0x7;
}

#endif

void __attribute__((noinline)) Application(void) {
    WAYPOINT("I");

    // Not using do_crt1 since it is copying to registers???
    // bss already cleared in entry code.
    // TODO: need to find free space that routing FW is not using

    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    noc_bank_table_init(eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SCRATCH);

    risc_init();
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    WAYPOINT("REW");
    uint32_t count = 0;
    while (routing_info->routing_enabled != 1) {
        volatile uint32_t* ptr = (volatile uint32_t*)0xffb2010c;
        count++;
        *ptr = 0xAABB0000 | (count & 0xFFFF);
        internal_::risc_context_switch();
    }
    WAYPOINT("RED");

    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0
    DeviceProfilerInit();
    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        uint8_t go_message_signal = mailboxes->go_messages[0].signal;
        if (go_message_signal == RUN_MSG_GO) {
            // Only include this iteration in the device profile if the launch message is valid. This is because all
            // workers get a go signal regardless of whether they're running a kernel or not. We don't want to profile
            // "invalid" iterations.
            DeviceZoneScopedMainN("ERISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
            // Note that a core may get "GO" w/ enable false to keep its launch_msg's in sync
            uint32_t enables = launch_msg_address->kernel_config.enables;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;
            if (enables & (1u << static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM0))) {
                WAYPOINT("R");
                firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, PROCESSOR_INDEX);
#if defined(ARCH_WORMHOLE) && defined(ENABLE_IRAM)
                iram_setup();
#endif
                extern void kernel_init();
                kernel_init();
                WAYPOINT("D");
            }
            mailboxes->go_messages[0].signal = RUN_MSG_DONE;

            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[0]);
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                internal_::notify_dispatch_core_done(dispatch_addr);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
                // Only executed if watcher is enabled. Ensures that we don't report stale data due to invalid launch
                // messages in the ring buffer
            }

        } else if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
            // Reset the launch message buffer read ptr
            mailboxes->launch_msg_rd_ptr = 0;
            if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                DeviceIncrementTraceCount();
                DeviceTraceOnlyProfilerInit();
            }
            uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[0]);
            mailboxes->go_messages[0].signal = RUN_MSG_DONE;
            internal_::notify_dispatch_core_done(dispatch_addr);
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}

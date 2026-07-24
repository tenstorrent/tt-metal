// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/firmware_common.h"
#include "internal/risc_attribs.h"
#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "internal/debug/sanitize.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h"
#include "hostdev/dev_msgs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/kernel_thread_globals.h"

uint8_t noc_index;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;

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

thread_local CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

thread_local uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
thread_local uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
thread_local uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
thread_local uint32_t rta_count __attribute__((used));
thread_local uint32_t crta_count __attribute__((used));
#endif

bank_noc_xy_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
bank_noc_xy_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(UNCACHED_MEM_MAILBOX_BASE);
tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;

inline void invalidate_kernel_binary_l2_cache(uintptr_t kernel_lma, launch_msg_t* launch_msg, uint32_t processor_index) {
    uint32_t kernel_size = launch_msg->kernel_config.kernel_text_size[processor_index];
    if (kernel_size == 0) {
        return;
    }
    invalidate_l2_cache_range(kernel_lma, kernel_size);
}

void device_setup() { setup_isr_csrs(); }

inline __attribute__((always_inline)) void signal_subordinate_completion() {
    uint32_t hartid = internal_::get_hw_thread_idx();
    *((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) = RUN_SYNC_MSG_DONE;
}

inline void start_subordinate_kernel_run_early(uint32_t enables) {
    for (int i = 1; i < NUM_DM_CORES; i++) {
        if (enables & (1u << i)) {
            *((volatile uint8_t*)&(subordinate_sync->dm1) + i - 1) = RUN_SYNC_MSG_GO;
        }
    }
}

inline void wait_subordinates() {
    WAYPOINT("NTW");
    subordinate_sync->padding = 0;
    while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE);
    WAYPOINT("NTD");
}

thread_local LocalDFBInterface g_dfb_interface[dfb::NUM_DFBS] __attribute__((used));
overlay::RemapperAPI g_remapper_configurator __attribute__((used));
volatile TxnDFBDescriptor g_txn_dfb_descriptor[32] __attribute__((used));
volatile KernelBarrier g_kernel_barrier[NUM_KERNEL_BARRIERS] __attribute__((used));

extern "C" uint32_t _start1() {
    configure_csr();
    uint32_t hartid = internal_::get_hw_thread_idx();
    if (hartid == 0) {
        extern uint32_t __ldm_data_start[];
        do_crt1(__ldm_data_start);
        (*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[hartid] = SHARED_GLOBALS_READY_GO;
    }
    extern uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    while ((*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[0] != SHARED_GLOBALS_READY_GO) {
    }
    WAYPOINT("I");
    DPRINT("DISPATCH DM0-FW: initialized\n");

    mailboxes->launch_msg_rd_ptr = 0;
    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    device_setup();
    if (hartid > 0) {
        signal_subordinate_completion();
    } else {
        risc_init();
        // Host-populated bank tables live in cached TL1; drop stale L2 lines before the copy.
        noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
        thread_sync_init();
        wait_subordinates();
        mailboxes->go_messages[0].signal = RUN_MSG_DONE;

        noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
        DeviceProfilerInit();
        while (1) {
            WAYPOINT("GW");
            uint8_t go_message_signal = RUN_MSG_DONE;
            DPRINT("DISPATCH DM0-FW: waiting for GO message\n");
            while (((go_message_signal = mailboxes->go_messages[mailboxes->go_message_index].signal) != RUN_MSG_GO) &&
                   !(mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.preload &
                     DISPATCH_ENABLE_FLAG_PRELOAD)) {
                if ((go_message_signal == RUN_MSG_RESET_READ_PTR) ||
                    (go_message_signal == RUN_MSG_RESET_READ_PTR_FROM_HOST) ||
                    (go_message_signal == RUN_MSG_REPLAY_TRACE)) {
                    mailboxes->launch_msg_rd_ptr = 0;
                    if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
                        if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                            DeviceIncrementTraceCount();
                            DeviceTraceOnlyProfilerInit();
                        }
                        uint32_t go_message_index = mailboxes->go_message_index;
                        uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                        mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;
                        DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                        notify_dispatch_core_done(dispatch_addr, noc_index);
                    }
                }
            }

            WAYPOINT("GD");

            {
                DeviceZoneScopedMainN("DM0-FW");
                uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
                launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
                DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
                DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
                uint32_t enables = launch_msg_address->kernel_config.enables;
                uintptr_t kernel_config_base =
                    firmware_config_init(mailboxes, ProgrammableCoreType::DISPATCH, hartid);

                for (uint32_t i = 0; i < MaxNumKernels; i++) {
                    mailboxes->shared_globals_ready[i] = SHARED_GLOBALS_READY_WAIT;
                }

                my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
                my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;
                overlay_cmd_buff_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);

                uint32_t tt_l1_ptr* dfb_l1_base =
                    (uint32_t tt_l1_ptr*)(MEM_L1_UNCACHED_BASE + kernel_config_base +
                                          launch_msg_address->kernel_config.local_cb_offset);
                start_subordinate_kernel_run_early(enables);

                uint32_t num_local_dfbs = launch_msg_address->kernel_config.local_cb_mask;
                setup_local_dfb_interfaces(dfb_l1_base, num_local_dfbs);

                int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM0);
                WAYPOINT("R");
                if (enables & (1u << index)) {
                    uintptr_t kernel_lma = launch_msg_address->kernel_config.kernel_text_offset[index];
                    invalidate_kernel_binary_l2_cache(kernel_lma, launch_msg_address, index);
                    invalidate_l1_icache();
                    auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                    record_stack_usage(stack_free);
                } else {
                    wait_for_go_message();
                }
                WAYPOINT("D");

                wait_subordinates();

                if (g_remapper_configurator.is_remapper_enabled()) {
                    g_remapper_configurator.clear_all_pairs();
                    g_remapper_configurator.disable_remapper();
                }

                uint32_t go_message_index = mailboxes->go_message_index;
                mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;

                if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                    launch_msg_address->kernel_config.enables = 0;
                    launch_msg_address->kernel_config.preload = 0;
                    uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                    DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                    CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                    notify_dispatch_core_done(dispatch_addr, noc_index);
                    mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
                }
            }
        }
    }

    while (1) {
        WAYPOINT("W1");
        while (true) {
            if (*((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) == RUN_SYNC_MSG_GO ||
                *((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) == RUN_SYNC_MSG_LOAD) {
                break;
            }
            asm("nop; nop; nop; nop; nop");
        }
        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

        uintptr_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::DISPATCH, hartid);
        int index = hartid;

        uintptr_t kernel_lma = launch_msg->kernel_config.kernel_text_offset[index];

        uint32_t tt_l1_ptr* dfb_l1_base = (uint32_t tt_l1_ptr*)(MEM_L1_UNCACHED_BASE + kernel_config_base +
                                                                launch_msg->kernel_config.local_cb_offset);
        uint32_t num_local_dfbs = launch_msg->kernel_config.local_cb_mask;

        setup_local_dfb_interfaces(dfb_l1_base, num_local_dfbs);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;
        overlay_cmd_buff_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);

        WAYPOINT("R1");
        while (*((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) != RUN_SYNC_MSG_GO) {
            asm("nop; nop; nop; nop; nop");
        }
        invalidate_kernel_binary_l2_cache(kernel_lma, launch_msg, index);
        invalidate_l1_icache();
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();

        record_stack_usage(stack_free);
        WAYPOINT("D1");
        DEVICE_PRINT_KERNEL_FINISHED();

        *((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) = RUN_SYNC_MSG_DONE;
    }

    return 0;
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/firmware_common.h"
// #include "risc_common.h"
#include "internal/risc_attribs.h"
#include "internal/debug/watcher_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "internal/debug/sanitize.h"
#include "tools/profiler/kernel_profiler.hpp"

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

thread_local uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
thread_local uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(UNCACHED_MEM_MAILBOX_BASE);
tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;

void device_setup() {
    // instn_buf
    // pc_buf
    // clock gating
    // NOC setup
    // set_deassert_addresses
    // wzeromem
    // invalidate_l1_cache
    // clear_destination_registers
    // enable_cc_stack
    // set_default_sfpu_constant_register_state
}

inline __attribute__((always_inline)) void signal_subordinate_completion() {
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    *((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) = RUN_SYNC_MSG_DONE;
}

inline void run_triscs(uint32_t enables) {
    // Wait for init_sync_registers to complete. Should always be done by the time we get here.
    // while (mailboxes->subordinate_sync.trisc0 != RUN_SYNC_MSG_DONE) {
    //     invalidate_l1_cache();
    // }

    // if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::MATH0)))
    // {
    //     mailboxes->subordinate_sync.trisc0 = RUN_SYNC_MSG_GO;
    //     mailboxes->subordinate_sync.trisc1 = RUN_SYNC_MSG_GO;
    //     mailboxes->subordinate_sync.trisc2 = RUN_SYNC_MSG_GO;
    // }
}

inline void start_subordinate_kernel_run_early(uint32_t enables) {
    for (int i = 1; i < NUM_DM_CORES; i++) {  // start from 1 to skip DM0
        if (enables & (1u << i)) {
            *((volatile uint8_t*)&(subordinate_sync->dm1) + i - 1) = RUN_SYNC_MSG_GO;
        }
    }
}

inline void wait_subordinates() {
    WAYPOINT("NTW");
    while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE);
    WAYPOINT("NTD");
}

// inline void trigger_sync_register_init() { mailboxes->subordinate_sync.trisc0 = RUN_SYNC_MSG_INIT_SYNC_REGISTERS; }

extern "C" uint32_t _start1() {
    configure_csr();
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    if (hartid == 0) {
        extern uint32_t __ldm_data_start[];
        do_crt1(__ldm_data_start);
    }
    extern uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    WAYPOINT("I");
    // handle noc_tobank ???
    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0
    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    // risc_init();
    device_setup();
    if (hartid > 0) {
        signal_subordinate_completion();
    } else {  // This is DM0
        noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);

        wait_subordinates();
        mailboxes->go_messages[0].signal = RUN_MSG_DONE;

        // trigger_sync_register_init();

        DeviceProfilerInit();
        while (1) {
            WAYPOINT("GW");
            uint8_t go_message_signal = RUN_MSG_DONE;
            // kernel_configs.preload is last in the launch message. so other data is
            // valid by the time it's set. All multicast data from the dispatcher is
            // written in order, so it will arrive in order. We also have a barrier
            // before mcasting the launch message (as a hang workaround), which
            // ensures that the unicast data will also have been received.
            while (((go_message_signal = mailboxes->go_messages[mailboxes->go_message_index].signal) != RUN_MSG_GO) &&
                   !(mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.preload &
                     DISPATCH_ENABLE_FLAG_PRELOAD)) {
                invalidate_l1_cache();
                // While the go signal for kernel execution is not sent, check if the worker was signalled
                // to reset its launch message read pointer.
                if ((go_message_signal == RUN_MSG_RESET_READ_PTR) ||
                    (go_message_signal == RUN_MSG_RESET_READ_PTR_FROM_HOST) ||
                    (go_message_signal == RUN_MSG_REPLAY_TRACE)) {
                    // Set the rd_ptr on workers to specified value
                    mailboxes->launch_msg_rd_ptr = 0;
                    if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
                        if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                            DeviceIncrementTraceCount();
                            DeviceTraceOnlyProfilerInit();
                        }
                        uint32_t go_message_index = mailboxes->go_message_index;
                        // Querying the noc_index is safe here, since the RUN_MSG_RESET_READ_PTR go signal is currently
                        // guaranteed to only be seen after a RUN_MSG_GO signal, which will set the noc_index to a valid
                        // value. For future proofing, the noc_index value is initialized to 0, to ensure an invalid NOC
                        // txn is not issued.
                        uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                        mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;
                        // Notify dispatcher that this has been done
                        DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                        notify_dispatch_core_done(dispatch_addr, noc_index);
                    }
                }
            }

            WAYPOINT("GD");

            {
                // Only include this iteration in the device profile if the launch message is valid. This is because all
                // workers get a go signal regardless of whether they're running a kernel or not. We don't want to
                // profile "invalid" iterations.
                DeviceZoneScopedMainN("DM0-FW");
                uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
                launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
                DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
                DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
                uint32_t enables = launch_msg_address->kernel_config.enables;
                // Trigger the NCRISC to start loading CBs and IRAM as soon as possible.
                // if (enables &
                //     (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1)))
                //     { subordinate_sync.dm1 = RUN_SYNC_MSG_LOAD;
                // }
                // Copies from L1 to IRAM on chips where NCRISC has IRAM
                uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, hartid);
                // Invalidate the i$ now the kernels have loaded and before running
                // volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
                // cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] =
                //     RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

                // run_triscs(enables);

                // noc_index = launch_msg_address->kernel_config.brisc_noc_id;
                // noc_mode = launch_msg_address->kernel_config.brisc_noc_mode;
                my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
                my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;
                noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                // re-initialize the NoCs
                // uint8_t cmd_buf;
                // if (noc_mode == DM_DEDICATED_NOC) {
                //     if (prev_noc_mode != noc_mode) {
                //         noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                //     }
                //     cmd_buf = BRISC_AT_CMD_BUF;
                // } else {
                //     if (prev_noc_mode != noc_mode) {
                //         dynamic_noc_init();
                //     }
                //     dynamic_noc_local_state_init();
                //     cmd_buf = DYNAMIC_NOC_BRISC_AT_CMD_BUF;
                // }
                // prev_noc_mode = noc_mode;

                // uint32_t tt_l1_ptr* cb_l1_base =
                //     (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.local_cb_offset);
                start_subordinate_kernel_run_early(enables);

                // Run the kernel
                WAYPOINT("R");
                int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM0);
                if (enables & (1u << index)) {
                    uint32_t local_cb_mask = launch_msg_address->kernel_config.local_cb_mask;
                    // TODO: setup DataFlowBuffers
                    // setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);
                    // cb_l1_base =
                    //     (uint32_t tt_l1_ptr*)(kernel_config_base +
                    //     launch_msg_address->kernel_config.remote_cb_offset);
                    // uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                    // experimental::setup_remote_cb_interfaces<true>(
                    //     cb_l1_base, end_cb_index, noc_index, noc_mode, true, cmd_buf);
                    // barrier_remote_cb_interface_setup(noc_index, end_cb_index);
                    uint32_t kernel_lma =
                        (kernel_config_base + launch_msg_address->kernel_config.kernel_text_offset[index]);
                    asm("FENCE.i");
                    auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                    record_stack_usage(stack_free);
                } else {
#if defined(PROFILE_KERNEL)
                    // This was not initialized in the kernel
                    // Currently FW does not issue a barrier except when using profiler
                    // if (noc_mode == DM_DEDICATED_NOC) {
                    //     noc_local_state_init(noc_index);
                    // }
#endif
                    // DM0 is responsible for issuing any noc cmds needed when initializing remote cbs
                    // So have DM0 setup remote cb interfaces even when DM0 is not in use
                    // if (launch_msg_address->kernel_config.enables) {
                    //     cb_l1_base =
                    //         (uint32_t tt_l1_ptr*)(kernel_config_base +
                    //         launch_msg_address->kernel_config.remote_cb_offset);
                    //     uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                    //     experimental::setup_remote_cb_interfaces<true>(
                    //         cb_l1_base, end_cb_index, noc_index, noc_mode, true, cmd_buf);
                    //     barrier_remote_cb_interface_setup(noc_index, end_cb_index);
                    // }
                    wait_for_go_message();
                }
                WAYPOINT("D");

                wait_subordinates();

                // trigger_sync_register_init();

                if constexpr (ASSERT_ENABLED) {
                    if (noc_mode == DM_DYNAMIC_NOC) {
                        WAYPOINT("NKFW");
                        // Assert that no noc transactions are outstanding, to ensure that all reads and writes have
                        // landed and the NOC interface is in a known idle state for the next kernel.
                        for (int noc = 0; noc < NUM_NOCS; noc++) {
                            ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
                            ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
                            ASSERT(ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
                            ASSERT(ncrisc_dynamic_noc_nonposted_atomics_flushed(noc));
                            ASSERT(ncrisc_dynamic_noc_posted_writes_sent(noc));
                        }
                        WAYPOINT("NKFD");
                    }
                }

#if defined(PROFILE_KERNEL)
                if (noc_mode == DM_DYNAMIC_NOC) {
                    // re-init for profiler to able to run barrier in dedicated noc mode
                    noc_local_state_init(noc_index);
                }
#endif

                uint32_t go_message_index = mailboxes->go_message_index;
                mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;

                // Notify dispatcher core that tensix has completed running kernels, if the launch_msg was populated
                if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                    // Set launch message to invalid, so that the next time this slot is encountered, kernels are only
                    // run if a valid launch message is sent.
                    launch_msg_address->kernel_config.enables = 0;
                    launch_msg_address->kernel_config.preload = 0;
                    uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                    DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                    // Only executed if watcher is enabled. Ensures that we don't report stale data due to invalid
                    // launch messages in the ring buffer. Must be executed before the atomic increment, as after that
                    // the launch message is no longer owned by us.
                    CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                    notify_dispatch_core_done(dispatch_addr, noc_index);
                    mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
                }
            }
        }
    }
    // Subordinates run this
    while (1) {
        // WAYPOINT("GW");
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

        uintptr_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, hartid);
        int index = hartid;

        uint32_t kernel_lma = kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index];

        uint32_t tt_l1_ptr* cb_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
        uint32_t local_cb_mask = launch_msg->kernel_config.local_cb_mask;
        // setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);

        // cb_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.remote_cb_offset);
        // uint32_t end_cb_index = launch_msg->kernel_config.min_remote_cb_start_index;
        // NOC argument is unused
        // experimental::setup_remote_cb_interfaces<false>(cb_l1_base, end_cb_index, 0, 0, 0, 0);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

        WAYPOINT("R1");
        while (*((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) != RUN_SYNC_MSG_GO) {
            asm("nop; nop; nop; nop; nop");
        }
        asm("FENCE.i");
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();

        record_stack_usage(stack_free);
        WAYPOINT("D1");

        *((volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1) = RUN_SYNC_MSG_DONE;
    }

    return 0;
}

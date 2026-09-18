// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/firmware_common.h"
#include "internal/risc_attribs.h"
#include "internal/hw_thread.h"
#include "internal/tt-2xx/risc_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/ring_buffer.h"
#include "hostdev/dev_msgs.h"

#include <cstddef>

uint8_t noc_index;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));

thread_local uint32_t hw_thread_idx __attribute__((used));
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

uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)] __attribute__((used));
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)] __attribute__((used));

// Uncached SRAM alias so hart 0's .data init and subordinate_sync are visible to the other harts.
tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_DRISC_MAILBOX_BASE);
tt_l1_ptr subordinate_map_t* const subordinate_sync =
    (subordinate_map_t*)(MEM_DRISC_MAILBOX_BASE + offsetof(mailboxes_t, subordinate_sync));

inline void invalidate_kernel_binary_l2_cache(
    uintptr_t kernel_lma, launch_msg_t* launch_msg, uint32_t processor_index) {
    uint32_t kernel_size = launch_msg->kernel_config.kernel_text_size[processor_index];
    if (kernel_size == 0) {
        return;
    }
    invalidate_l2_cache_range(kernel_lma, kernel_size);
}

inline __attribute__((always_inline)) volatile uint8_t* subordinate_sync_slot(uint32_t hartid) {
    return (volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1;
}

inline __attribute__((always_inline)) void signal_subordinate_completion() {
    uint32_t hartid = internal_::get_hw_thread_idx();
    *subordinate_sync_slot(hartid) = RUN_SYNC_MSG_DONE;
}

inline void start_subordinate_kernel_run_early(uint32_t enables) __attribute__((used));
inline void start_subordinate_kernel_run_early(uint32_t enables) {
    for (uint32_t i = 1; i < MEM_CCE_LOCAL_HARTS; i++) {
        if (enables & (1u << i)) {
            *subordinate_sync_slot(i) = RUN_SYNC_MSG_GO;
        }
    }
}

inline void wait_subordinates() {
    WAYPOINT("NTW");
    subordinate_sync->padding = 0;
    while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE) {
    }
    WAYPOINT("NTD");
}

extern "C" uint32_t _start1() {
    // Raw read: hw_thread_idx has not been filled yet, and do_thread_crt1() below zeroes the .tbss
    // it lives in, so caching it any earlier would just be discarded.
    uint32_t hartid = internal_::read_hw_thread_idx();
    configure_csr();
    if (hartid == 0) {
        extern uint32_t __ldm_data_start[];
        do_crt1(__ldm_data_start);
        // Must precede the ready flag below, which releases the other harts.
        WATCHER_RING_BUFFER_INIT();
        (*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[hartid] = SHARED_GLOBALS_READY_GO;
    }
    extern uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    internal_::init_hw_thread_idx();
    while ((*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[0] != SHARED_GLOBALS_READY_GO) {
    }
    WAYPOINT("I");

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    noc_index = 0;

    if (hartid > 0) {
        signal_subordinate_completion();
        while (1) {
            WAYPOINT("W1");
            while (*subordinate_sync_slot(hartid) != RUN_SYNC_MSG_GO &&
                   *subordinate_sync_slot(hartid) != RUN_SYNC_MSG_LOAD) {
                asm("nop; nop; nop; nop; nop");
            }
            while (*subordinate_sync_slot(hartid) != RUN_SYNC_MSG_GO) {
                asm("nop; nop; nop; nop; nop");
            }
            WAYPOINT("R1");
            // Host DRAM kernels are a single processor (hart 0). Subordinates stay in the GO/DONE
            // handshake until per-hart CCE kernels exist.
            WAYPOINT("D1");
            *subordinate_sync_slot(hartid) = RUN_SYNC_MSG_DONE;
        }
    }

    risc_init();
    noc_bank_table_init(MEM_CCE_BANK_TO_NOC_SCRATCH);
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    mailboxes->launch_msg_rd_ptr = 0;
    wait_subordinates();
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;

    while (1) {
        WAYPOINT("GW");
        while (mailboxes->go_messages[0].signal != RUN_MSG_GO) {
        }
        WAYPOINT("GD");

        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &mailboxes->launch[launch_msg_rd_ptr];
        uint32_t enables = launch_msg->kernel_config.enables;
        firmware_config_init(mailboxes, ProgrammableCoreType::DRAM, hartid);
        start_subordinate_kernel_run_early(enables);
        overlay_cmd_buff_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);

        WAYPOINT("R");
        if (enables & 1u) {
            uintptr_t kernel_lma = launch_msg->kernel_config.kernel_text_offset[0];
            // Invalidate the i$ now the kernels have loaded and before running
            invalidate_kernel_binary_l2_cache(kernel_lma, launch_msg, 0);
            invalidate_l1_icache();
            reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        }
        WAYPOINT("D");

        wait_subordinates();
        mailboxes->go_messages[0].signal = RUN_MSG_DONE;
        if (launch_msg->kernel_config.mode == DISPATCH_MODE_DEV) {
            launch_msg->kernel_config.enables = 0;
            mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
        }
        WAYPOINT("GW");
    }

    return 0;
}

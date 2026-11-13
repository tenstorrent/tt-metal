// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/firmware_common.h"
#include "internal/risc_attribs.h"
#include "internal/hw_thread.h"
#include "internal/tt-2xx/risc_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/ring_buffer.h"
#include "api/debug/device_print.h"
#include "hostdev/dev_msgs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
#include "internal/tt-2xx/quasar/cache.h"
#endif

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

// One image per CCE: CCE0's is linked at local 0, CCE1's MEM_IERISC_ETH_CCE_BASE(1) up. Both reach the
// shared mailbox in CCE0's SRAM at the same local address. Processor 0 (CCE0 hart 0) is the master
// for all sixteen harts; every other hart, CCE1's included, is a subordinate.
// Uncached alias. L2 does not snoop the host's NOC->TL1 writes, so a cached read of the go
// message would spin on the stale line the previous iteration left behind.
inline tt_l1_ptr mailboxes_t* ierisc_mailboxes() {
    return (tt_l1_ptr mailboxes_t*)(MEM_IERISC_MAILBOX_BASE + MEM_L1_UNCACHED_BASE);
}

inline tt_l1_ptr subordinate_map_t* ierisc_subordinate_sync() {
    return (tt_l1_ptr subordinate_map_t*)(MEM_IERISC_MAILBOX_BASE + MEM_L1_UNCACHED_BASE +
                                          offsetof(mailboxes_t, subordinate_sync));
}

#define mailboxes (ierisc_mailboxes())
#define subordinate_sync (ierisc_subordinate_sync())

// One byte per subordinate, processors 1-15: the DM bytes, the padding byte and the first Neo bytes of
// subordinate_map_t (an ethernet core has no Neo). Byte 15 has no processor and stays 0.
inline __attribute__((always_inline)) volatile uint8_t* subordinate_sync_slot(uint32_t hartid) {
    return (volatile uint8_t*)&(subordinate_sync->dm1) + hartid - 1;
}

inline __attribute__((always_inline)) bool all_subordinates_done() {
    volatile uint64_t* words = (volatile uint64_t*)subordinate_sync;
    return words[0] == RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE && words[1] == RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE;
}

inline __attribute__((always_inline)) void signal_subordinate_completion() {
    uint32_t hartid = internal_::get_hw_thread_idx();
    *subordinate_sync_slot(hartid) = RUN_SYNC_MSG_DONE;
}

inline void invalidate_kernel_binary_l2_cache(
    uintptr_t kernel_lma, launch_msg_t* launch_msg, uint32_t processor_index) {
    uint32_t kernel_size = launch_msg->kernel_config.kernel_text_size[processor_index];
    if (kernel_size == 0) {
        return;
    }
    invalidate_l2_cache_range(kernel_lma, kernel_size);
}

inline void start_subordinate_kernel_run_early(uint32_t enables) __attribute__((used));
inline void start_subordinate_kernel_run_early(uint32_t enables) {
    for (uint32_t i = 1; i < PROCESSOR_COUNT; i++) {
        if (enables & (1u << i)) {
            *subordinate_sync_slot(i) = RUN_SYNC_MSG_GO;
        }
    }
}

inline void wait_subordinates() {
    WAYPOINT("NTW");
    while (!all_subordinates_done()) {
    }
    WAYPOINT("NTD");
}

extern "C" uint32_t _start1() {
    // Raw read: hw_thread_idx has not been filled yet, and do_thread_crt1() below zeroes the .tbss
    // it lives in, so caching it any earlier would just be discarded.
    uint32_t hartid = internal_::read_hw_thread_idx();
    configure_csr();
    // Hart 0 of each CCE initializes that CCE's image. Processor 0 also owns the shared mailbox state.
    constexpr uint32_t image_leader = ETH_PHYSICAL_CCE_ID;
    if (hartid == image_leader) {
        extern uint32_t __ldm_data_start[];
        do_crt1(__ldm_data_start);
        if (hartid == 0) {
            WATCHER_RING_BUFFER_INIT();
            DEVICE_PRINT_INITIALIZE_LOCK();
        }
        risc_init();
        mailboxes->fw_shared_globals_ready[image_leader] = SHARED_GLOBALS_READY_GO;
    }
    extern uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    internal_::init_hw_thread_idx();
    while (mailboxes->fw_shared_globals_ready[image_leader] != SHARED_GLOBALS_READY_GO) {
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
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg = &mailboxes->launch[launch_msg_rd_ptr];
            firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, hartid);
            WAYPOINT("R1");
            while (*subordinate_sync_slot(hartid) != RUN_SYNC_MSG_GO) {
                asm("nop; nop; nop; nop; nop");
            }
            uintptr_t kernel_lma = launch_msg->kernel_config.kernel_text_offset[hartid];
            invalidate_kernel_binary_l2_cache(kernel_lma, launch_msg, hartid);
            WAYPOINT("L1");
            invalidate_l1_icache();
            WAYPOINT("I1");
            reinterpret_cast<uint32_t (*)()>(kernel_lma)();
            WAYPOINT("D1");
            DEVICE_PRINT_KERNEL_FINISHED();
            *subordinate_sync_slot(hartid) = RUN_SYNC_MSG_DONE;
        }
    }

    mailboxes->launch_msg_rd_ptr = 0;
    wait_subordinates();
    mailboxes->go_messages[0].slot[0].signal = RUN_MSG_DONE;

    while (1) {
        WAYPOINT("GW");
        while (mailboxes->go_messages[0].slot[0].signal != RUN_MSG_GO) {
        }
        WAYPOINT("GD");

        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &mailboxes->launch[launch_msg_rd_ptr];
        uint32_t enables = launch_msg->kernel_config.enables;
        firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, hartid);
        start_subordinate_kernel_run_early(enables);

        // A stale GO from the previous launch would release subordinate harts before thread 0 finishes CRT.
        for (uint32_t i = 0; i < MaxNumKernels; i++) {
            mailboxes->shared_globals_ready[i] = SHARED_GLOBALS_READY_WAIT;
        }

        WAYPOINT("R");
        if (enables & (1u << hartid)) {
            uintptr_t kernel_lma = launch_msg->kernel_config.kernel_text_offset[hartid];
            invalidate_kernel_binary_l2_cache(kernel_lma, launch_msg, hartid);
            WAYPOINT("RL");
            invalidate_l1_icache();
            WAYPOINT("RI");
            reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        }
        WAYPOINT("D");
        DEVICE_PRINT_KERNEL_FINISHED();

        wait_subordinates();
        mailboxes->go_messages[0].slot[0].signal = RUN_MSG_DONE;
        if (launch_msg->kernel_config.mode == DISPATCH_MODE_DEV) {
            launch_msg->kernel_config.enables = 0;
            mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
        }
        WAYPOINT("GW");
    }

    return 0;
}

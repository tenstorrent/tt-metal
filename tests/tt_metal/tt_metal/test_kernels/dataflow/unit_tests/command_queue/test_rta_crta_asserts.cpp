// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher RTA/CRTA bounds checking:
// 1. Writes rta_count, crta_count, and all arg values to L1 for validation
// 2. If MAX_RTA_IDX/MAX_CRTA_IDX defined: accesses that index to test bounds checking
// Supports both DM (BRISC/NCRISC/DM0-7) and compute (TRISC0-3) kernels

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#include "internal/firmware_common.h"
#else
#include "api/compute/common.h"
#endif

#ifdef ARCH_QUASAR
// TODO: Remove this once cache invalidation functionality for Quasar is added
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
inline __attribute__((always_inline)) void flush_l2_cache_line(uint32_t* addr) {
    asm volatile("fence" ::: "memory");
    volatile uint64_t* flush_reg = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
    *flush_reg = reinterpret_cast<uint64_t>(addr);
    asm volatile("fence" ::: "memory");
}

thread_local extern uint32_t rta_count;
thread_local extern uint32_t crta_count;
#else
extern uint32_t rta_count;
extern uint32_t crta_count;
#endif

// Helper to write RTA/CRTA metadata and values to L1
static FORCE_INLINE void write_args_to_l1(uint32_t l1_write_addr) {
    uint32_t* ptr = reinterpret_cast<uint32_t*>(l1_write_addr);
    ptr[0] = rta_count;
    ptr[1] = crta_count;

    for (size_t i = 0; i < rta_count; i++) {
        ptr[i + 2] = get_arg_val<uint32_t>(i);
    }
    for (size_t i = 0; i < crta_count; i++) {
        ptr[i + rta_count + 2] = get_common_arg_val<uint32_t>(i);
    }

#ifdef ARCH_QUASAR
    flush_l2_cache_line(ptr);
#endif  // ARCH_QUASAR
}

#ifndef COMPILE_FOR_TRISC

void core_agnostic_main() {
#if defined(COMPILE_FOR_DM)
    uint32_t thread_idx = get_my_thread_id();

#if defined(TEST_MULTI_DM_RTA)
    // Multi-DM mode: sync barrier so all DMs hit the OOB access together,
    // stress-testing watcher's first-writer-wins assert mechanism.
    // Compile args: [num_dms, l1_sync_addr, l1_scratch_addr]
    constexpr uint32_t num_dms = get_compile_time_arg_val(0);
    constexpr uint32_t l1_sync_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_scratch_addr = get_compile_time_arg_val(2);
    uint64_t* l1_ptr = reinterpret_cast<uint64_t*>(l1_sync_addr);
    __atomic_add_fetch(l1_ptr, 1, __ATOMIC_RELAXED);
    while (__atomic_load_n(l1_ptr, __ATOMIC_ACQUIRE) != num_dms) {
    }
#else
    // Single DM test: only specified dm_id executes, others exit early
    // Compile args: [dm_id, l1_scratch_addr]
    constexpr uint32_t dm_id = get_compile_time_arg_val(0);
    constexpr uint32_t l1_scratch_addr = get_compile_time_arg_val(1);
    if (thread_idx != dm_id) {
        return;
    }
#endif  // TEST_MULTI_DM_RTA
#else   // COMPILE_FOR_DM
    // Non-Quasar DM: L1 scratch address is the first compile-time arg
    constexpr uint32_t l1_scratch_addr = get_compile_time_arg_val(0);
#endif

    write_args_to_l1(l1_scratch_addr);
    DPRINT << "done write_args_to_l1" << ENDL();

#if defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
    // Signal completion to dispatcher before assert hangs the kernel
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);

    // SD signaling: IDLE_ERISC (all archs) and Quasar DM require RUN_MSG_DONE
    // TODO: Remove COMPILE_FOR_DM once FD is enabled on Quasar
#if defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_DM)
    go_message_in->signal = RUN_MSG_DONE;
#else
    // FD: ACTIVE_ETH, BRISC, NCRISC notify dispatcher via NOC
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif  // COMPILE_FOR_IDLE_ERISC or COMPILE_FOR_DM
#endif  // MAX_RTA_IDX or MAX_CRTA_IDX

#ifdef MAX_RTA_IDX
    // Access RTA: this should have a watcher assert when MAX_RTA_IDX >= rta_count
    uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif  // MAX_RTA_IDX
#ifdef MAX_CRTA_IDX
    // Access CRTA: this should have a watcher assert when MAX_CRTA_IDX >= crta_count
    uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif  // MAX_CRTA_IDX
    DPRINT << "Completed" << ENDL();
}

#else  // Compute Kernel

void core_agnostic_main() {
    UNPACK({
        // L1 scratch address passed as compile-time arg
        write_args_to_l1(get_compile_time_arg_val(0));

#if defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
        // Signal completion before triggering assert
#if defined(ARCH_QUASAR)
        // Quasar SD: signal via go_message
        volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
        go_message_in->signal = RUN_MSG_DONE;
#else
        // BH/WH: signal via subordinate sync
        volatile tt_l1_ptr mailboxes_t* mailbox = reinterpret_cast<volatile tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
        volatile tt_l1_ptr subordinate_map_t* sync =
            reinterpret_cast<volatile tt_l1_ptr subordinate_map_t*>(&mailbox->subordinate_sync);
        sync->trisc0 = RUN_SYNC_MSG_DONE;
#endif  // ARCH_QUASAR
#endif  // MAX_RTA_IDX or //MAX_CRTA_IDX

#ifdef MAX_RTA_IDX
        // Access RTA: this should have a watcher assert when MAX_RTA_IDX >= rta_count
        uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif  // MAX_RTA_IDX
#ifdef MAX_CRTA_IDX
        // Access CRTA: this should have a watcher assert when MAX_CRTA_IDX >= crta_count
        uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif  // MAX_CRTA_IDX
    })
}
#endif  // COMPILE_FOR_TRISC

void kernel_main() { core_agnostic_main(); }

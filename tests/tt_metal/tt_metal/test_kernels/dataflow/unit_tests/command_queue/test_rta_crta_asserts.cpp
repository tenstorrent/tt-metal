// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher RTA/CRTA bounds checking:
// 1. Writes rta_count, crta_count, and all arg values to L1 for validation
// 2. If MAX_RTA_IDX/MAX_CRTA_IDX defined: accesses that index to test bounds checking
// Supports both DM and compute kernels

#include "experimental/core_local_mem.h"
#include "api/kernel_thread_globals.h"

#ifndef COMPILE_FOR_TRISC
#include "internal/firmware_common.h"
#else
#include "api/compute/common.h"
#endif

#ifdef ARCH_QUASAR
// TODO: Remove this once PR #38124 is merged
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
inline __attribute__((always_inline)) void flush_l2_cache_line(uintptr_t addr) {
    asm volatile("fence" ::: "memory");
    volatile uint64_t* flush_reg = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
    *flush_reg = static_cast<uint64_t>(addr);
    asm volatile("fence" ::: "memory");
}

thread_local extern uint32_t rta_count;
thread_local extern uint32_t crta_count;
#else
extern uint32_t rta_count;
extern uint32_t crta_count;
#endif

// Helper: Signal completion to dispatcher before assert hangs the kernel
static FORCE_INLINE void signal_completion_before_assert() {
#if defined(ARCH_QUASAR)
    // Quasar SD: signal via go_message
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_in->signal = RUN_MSG_DONE;
#else  // Else WH/BH
#ifdef COMPILE_FOR_TRISC
    // signal via subordinate sync
    volatile tt_l1_ptr subordinate_map_t* sync =
        reinterpret_cast<volatile tt_l1_ptr subordinate_map_t*>(GET_MAILBOX_ADDRESS_DEV(subordinate_sync));
    sync->trisc0 = RUN_SYNC_MSG_DONE;
#else
    // FD: BRISC, NCRISC notify dispatcher via NOC
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif  // COMPILE_FOR_TRISC
#endif  // ARCH_QUASAR
}

// Helper: trigger bounds-check assert by accessing arg beyond bounds
static FORCE_INLINE void trigger_bounds_check_assert() {
#ifdef MAX_RTA_IDX
    volatile uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif
#ifdef MAX_CRTA_IDX
    volatile uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif
}

// Helper: write RTA/CRTA metadata and values to L1
static FORCE_INLINE void write_args_to_l1(uint32_t l1_write_addr) {
    experimental::CoreLocalMem<uint32_t> ptr(l1_write_addr);
    ptr[0] = rta_count;
    ptr[1] = crta_count;

    for (size_t i = 0; i < rta_count; i++) {
        ptr[i + 2] = get_arg_val<uint32_t>(i);
    }
    for (size_t i = 0; i < crta_count; i++) {
        ptr[i + rta_count + 2] = get_common_arg_val<uint32_t>(i);
    }

#ifdef ARCH_QUASAR
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(ptr.get_address()));
#endif
}

#ifndef COMPILE_FOR_TRISC

void core_agnostic_main() {
#if defined(COMPILE_FOR_DM)
    // Quasar DM kernel
    uint32_t thread_idx = get_my_thread_id();

#if defined(TEST_MULTI_DM_RTA)
    // Multi-DM mode: Spin-wait for all DMs to reach barrier so all hit the bounds-check access together
    // Compile args: [num_dms, l1_sync_addr]
    constexpr uint32_t num_dms = get_compile_time_arg_val(0);
    constexpr uint32_t l1_sync_addr = get_compile_time_arg_val(1);
    experimental::CoreLocalMem<uint32_t> l1_sync_ptr(l1_sync_addr);
    __atomic_add_fetch(l1_sync_ptr.get_unsafe_ptr(), 1, __ATOMIC_RELAXED);
    while (__atomic_load_n(l1_sync_ptr.get_unsafe_ptr(), __ATOMIC_ACQUIRE) != num_dms) {
    }
#elif defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
    // Assert test: only specified dm_id executes, others exit early
    // Compile args: [dm_id]
    constexpr uint32_t dm_id = get_compile_time_arg_val(0);
    if (thread_idx != dm_id) {
        return;
    }
#else
    // Validation test: write args to L1 for host readback
    // Compile args: [dm_id, l1_scratch_addr]
    constexpr uint32_t dm_id = get_compile_time_arg_val(0);
    constexpr uint32_t l1_scratch_addr = get_compile_time_arg_val(1);
    if (thread_idx != dm_id) {
        return;
    }
    write_args_to_l1(l1_scratch_addr);
#endif

#else  // Non-Quasar DM (BRISC/NCRISC)

#if !defined(MAX_RTA_IDX) && !defined(MAX_CRTA_IDX)
    // Validation test: L1 scratch address is the first compile-time arg
    constexpr uint32_t l1_scratch_addr = get_compile_time_arg_val(0);
    write_args_to_l1(l1_scratch_addr);
#endif

#endif

#if defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
    signal_completion_before_assert();
    trigger_bounds_check_assert();
#endif
}

#else  // Compute Kernel

void core_agnostic_main() {
    UNPACK({
#if !defined(MAX_RTA_IDX) && !defined(MAX_CRTA_IDX)
        write_args_to_l1(get_compile_time_arg_val(0));
#else
        signal_completion_before_assert();
        trigger_bounds_check_assert();
#endif
    })
}
#endif

void kernel_main() { core_agnostic_main(); }

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing RISCV atomics with L1 using gcc built-in Functions
// Kernel is only launched archs that support atomics: Quasar and BH
// While QSR supports both Zaamo and Zalrsc, BH only supports Zaamo
// On BH, the RISCs always execute Zaamo aq/rl variants as if both bits were set
// So relaxed/release/acquire/seq-cst collapse to the same hardware behavior,
// though ELF disassembly shows the gcc built-ins generate stronger code for seq-cst
// BH uses 32-bit atomics here; Quasar uses 64-bit atomics here

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/common.h"
#endif

#include "api/debug/dprint.h"

#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

typedef uint64_t atomic_type;

// TODO: Remove this once cache invalidation functionality for Quasar is added
inline __attribute__((always_inline)) void flush_l2_cache_line(atomic_type* addr) {
    asm volatile("fence" ::: "memory");
    volatile atomic_type* flush_reg = (volatile atomic_type*)L2_FLUSH_ADDR;
    *flush_reg = (atomic_type)addr;
    asm volatile("fence" ::: "memory");
}
#else
typedef uint32_t atomic_type;
#endif

// Writer publishes the shared value, then sets the ready flag
void writer(atomic_type* shared_value_ptr, atomic_type* ready_flag_ptr) {
    *shared_value_ptr = SENTINEL;
    __atomic_store_n(ready_flag_ptr, 1, __ATOMIC_SEQ_CST);
}

// Reader waits for the ready flag, then reads the published value
void reader(atomic_type* shared_value_ptr, atomic_type* ready_flag_ptr, atomic_type* result_ptr) {
    while (!__atomic_load_n(ready_flag_ptr, __ATOMIC_SEQ_CST)) {
    }
    *result_ptr = *shared_value_ptr;
}

void test_atomic_load_store(atomic_type* shared_value_ptr, atomic_type* ready_flag_ptr, atomic_type* result_ptr) {
#ifdef ARCH_QUASAR
    uint64_t thread_idx = 0;
    asm volatile("csrr %0, mhartid" : "=r"(thread_idx));
    // On Quasar, DM0 runs writer() and DM1-7 run reader()
    // Each DM writes to its own result location for host to verify
    if (thread_idx == 0) {
        writer(shared_value_ptr, ready_flag_ptr);
    } else {
        // Each reader DM writes to a unique result slot so host verification is per thread
        atomic_type* per_thread_result_ptr = result_ptr + (thread_idx - 1);
        reader(shared_value_ptr, ready_flag_ptr, per_thread_result_ptr);
        // Flush the cache so host readback sees the updated L1
        flush_l2_cache_line(per_thread_result_ptr);
    }
#else
    // ON BH, BRISC runs writer(), NCRISC runs reader()
#ifdef COMPILE_FOR_BRISC
    writer(shared_value_ptr, ready_flag_ptr);
#else
    reader(shared_value_ptr, ready_flag_ptr, result_ptr);
#endif  // COMPILE_FOR_BRISC
#endif  // ARCH_QUASAR
}

void test_atomic_add_fetch(atomic_type* l1_counter_ptr, const atomic_type increment_times) {
    for (uint32_t i = 0; i < increment_times; i++) {
        __atomic_add_fetch(l1_counter_ptr, 1, __ATOMIC_RELEASE);
    }
}

void test_compare_and_swap_atomic(atomic_type* l1_counter_ptr, const uint32_t increment_times) {
    for (uint32_t i = 0; i < increment_times; i++) {
        atomic_type expected_read = __atomic_load_n(l1_counter_ptr, __ATOMIC_SEQ_CST);
        while (!__atomic_compare_exchange_n(
            l1_counter_ptr, &expected_read, expected_read + 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        }
    }
}

void kernel_main() {
    // shared L1 location between all DMs for atomic updates
    atomic_type* l1_counter_ptr = reinterpret_cast<atomic_type*>(get_arg_val<uint32_t>(0));
    const uint32_t increment_times = get_arg_val<uint32_t>(1);

#if TEST_ATOMIC_LOAD_STORE
    atomic_type* shared_value_ptr = l1_counter_ptr;
    atomic_type* ready_flag_ptr = (l1_counter_ptr + 1);
    atomic_type* result_ptr = (l1_counter_ptr + 2);
    test_atomic_load_store(shared_value_ptr, ready_flag_ptr, result_ptr);
#elif TEST_ATOMIC_ADD_FETCH
    test_atomic_add_fetch(l1_counter_ptr, increment_times);
#elif TEST_ATOMIC_CAS && defined(ARCH_QUASAR)
    test_compare_and_swap_atomic(l1_counter_ptr, increment_times);
#endif

#if defined(ARCH_QUASAR)
    // Flush the cache so host readback sees the updated L1
    flush_l2_cache_line(l1_counter_ptr);
#endif
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing RISCV atomics with L1 using gcc built-in Functions
// Kernel is only launched on archs that support atomics: Quasar and BH
// While QSR supports both Zaamo and Zalrsc, BH only supports Zaamo
// On BH, the RISCs always execute Zaamo aq/rl variants as if both bits were set
// So relaxed/release/acquire/seq-cst collapse to the same hardware behavior,
// though ELF disassembly shows the gcc built-ins generate stronger code for seq-cst
// BH uses 32-bit atomics; Quasar uses 64-bit atomics

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/common.h"
#endif

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

#if TEST_ATOMIC_LOAD_STORE
// Writer atomically writes the sentinel value
void writer(atomic_type* shared_value_ptr) { __atomic_store_n(shared_value_ptr, SENTINEL, __ATOMIC_SEQ_CST); }

// Reader spins until atomic load sees SENTINEL, then writes the observed value for host verification
void reader(atomic_type* shared_value_ptr, atomic_type* result_ptr) {
    atomic_type val;
    while ((val = __atomic_load_n(shared_value_ptr, __ATOMIC_SEQ_CST)) != SENTINEL) {
    }
    *result_ptr = val;
}

void test_atomic_load_store(atomic_type* shared_value_ptr, atomic_type* result_ptr) {
#ifdef ARCH_QUASAR
    uint64_t thread_idx = 0;
    asm volatile("csrr %0, mhartid" : "=r"(thread_idx));
    // On Quasar, DM0 runs writer() and DM1-7 run reader()
    // Each DM writes to its own result location for host to verify
    if (thread_idx == 0) {
        writer(shared_value_ptr);
    } else {
        // Each reader DM writes to a unique result slot so host verification is per thread
        atomic_type* per_thread_result_ptr = result_ptr + (thread_idx - 1);
        reader(shared_value_ptr, per_thread_result_ptr);
        // Flush the cache so host readback sees the updated L1
        flush_l2_cache_line(per_thread_result_ptr);
    }
#else
    // ON BH, BRISC runs writer(), NCRISC runs reader()
#ifdef COMPILE_FOR_BRISC
    writer(shared_value_ptr);
#else
    reader(shared_value_ptr, result_ptr);
#endif  // COMPILE_FOR_BRISC
#endif  // ARCH_QUASAR
}
#endif  // TEST_ATOMIC_LOAD_STORE

#ifdef TEST_ATOMIC_ADD_FETCH
void test_atomic_add_fetch(atomic_type* l1_counter_ptr, const uint32_t increment_times) {
    for (uint32_t i = 0; i < increment_times; i++) {
        __atomic_add_fetch(l1_counter_ptr, 1, __ATOMIC_SEQ_CST);
    }
}
#endif  // TEST_ATOMIC_ADD_FETCH

#if defined(TEST_ATOMIC_CAS) && defined(ARCH_QUASAR)
void test_compare_and_swap_atomic(atomic_type* l1_counter_ptr, const uint32_t increment_times) {
    for (uint32_t i = 0; i < increment_times; i++) {
        atomic_type expected_read = __atomic_load_n(l1_counter_ptr, __ATOMIC_SEQ_CST);
        while (!__atomic_compare_exchange_n(
            l1_counter_ptr, &expected_read, expected_read + 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        }
    }
}
#endif  // TEST_ATOMIC_CAS

void kernel_main() {
    // Base L1 address shared by all DMs: used as a counter (add/CAS) or value + result slots (load/store)
    atomic_type* l1_counter_ptr = reinterpret_cast<atomic_type*>(get_arg_val<uint32_t>(0));
    const uint32_t increment_times = get_arg_val<uint32_t>(1);

#if TEST_ATOMIC_LOAD_STORE
    atomic_type* shared_value_ptr = l1_counter_ptr;
    atomic_type* result_ptr = (l1_counter_ptr + 1);
    test_atomic_load_store(shared_value_ptr, result_ptr);
#elif TEST_ATOMIC_ADD_FETCH
    test_atomic_add_fetch(l1_counter_ptr, increment_times);
#elif TEST_ATOMIC_CAS && defined(ARCH_QUASAR)
    test_compare_and_swap_atomic(l1_counter_ptr, increment_times);
#endif

#if defined(ARCH_QUASAR) && (defined(TEST_ATOMIC_ADD_FETCH) || defined(TEST_ATOMIC_CAS))
    // Flush the cache so host readback sees the updated L1
    flush_l2_cache_line(l1_counter_ptr);
#endif
}

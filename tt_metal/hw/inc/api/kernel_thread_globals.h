// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#if defined(ARCH_QUASAR)

// Per-processor kernel thread info, set by Quasar dm.cc/trisc.cc from kernel_config before kernel runs.
// Used by dmk.cc, trisc.cc, and runtime (e.g. CircularBuffers) via get_num_threads() / get_my_thread_id().
extern thread_local uint32_t num_sw_threads;
extern thread_local uint32_t my_thread_id;

#ifndef COMPILE_FOR_TRISC
/**
 * Reusable software barrier for DM kernel threads.
 *
 * Uses a generation-based barrier on Quasar, and is a
 * no-op on WH/BH where DM execution is single-threaded.
 */
struct KernelBarrier {
    uint32_t arrived = 0;
    uint32_t generation = 0;
};

// Per-side barrier state for DM kernels on a worker. A DFB's producer and consumer
// kernels can co-reside on one worker with different thread counts; a single shared
// barrier deadlocks because wait_threads() keys the release on the ARRIVING hart's own
// participant count against a shared counter, so mixed counts (e.g. producers=2,
// consumers=4) never hit a target for some arrival orders. Give the producer-side and
// consumer-side rendezvous separate barriers so each syncs its own threads.
//
// Invariant this relies on: at most one producer-role and one consumer-role multi-thread
// DM rendezvous group per worker. Two co-resident same-role multi-thread DM kernels with
// different thread counts would still share a slot (host validation admits at most one
// same-role DFB instance per node today, so this is not a reachable topology); if that
// ever becomes supported, key the barrier per kernel-group instead of the fixed 2 slots.
constexpr uint32_t NUM_KERNEL_BARRIERS = 2;  // [0] = producer side, [1] = consumer side
extern volatile KernelBarrier g_kernel_barrier[NUM_KERNEL_BARRIERS];

#endif // !COMPILE_FOR_TRISC

#endif // ARCH_QUASAR

// clang-format off
/**
 * Returns the number of threads (processors) in the kernel that this processor belongs to.
 * Set by Quasar firmware from kernel_config before the kernel runs. Valid only on ARCH_QUASAR.
 *
 * Return value: Number of kernel threads (num_processors_per_cluster for this kernel).
 */
// clang-format on
inline uint32_t get_num_threads() {
#if defined(ARCH_QUASAR)
    return num_sw_threads;
#else
    return 1;
#endif
}

// clang-format off
/**
 * Returns this processor's thread ID within its kernel (0 to get_num_threads() - 1).
 * Set by Quasar firmware from kernel_config before the kernel runs. Valid only on ARCH_QUASAR.
 *
 * Return value: Thread ID for this processor.
 */
// clang-format on
inline uint32_t get_my_thread_id() {
#if defined(ARCH_QUASAR)
    return my_thread_id;
#else
    return 0;
#endif
}

#ifndef COMPILE_FOR_TRISC

inline void thread_sync_init() {
#if defined(ARCH_QUASAR)
    for (uint32_t i = 0; i < NUM_KERNEL_BARRIERS; i++) {
        g_kernel_barrier[i].arrived = 0;
        g_kernel_barrier[i].generation = 0;
    }
#endif
}

// barrier_idx selects an independent barrier so co-resident kernels with different
// participant counts (e.g. a DFB's producer vs consumer kernel) don't share a counter.
inline void wait_threads(uint32_t participants, uint32_t barrier_idx = 0) {
    if (participants <= 1) {
        return;
    }

#if defined(ARCH_QUASAR)
    volatile KernelBarrier& barrier = g_kernel_barrier[barrier_idx];
    uint32_t next_generation = __atomic_load_n(&barrier.generation, __ATOMIC_ACQUIRE) + 1;
    uint32_t arrived = __atomic_add_fetch(&barrier.arrived, 1, __ATOMIC_ACQ_REL);
    if (arrived == participants) {
        __atomic_store_n(&barrier.arrived, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&barrier.generation, next_generation, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&barrier.generation, __ATOMIC_ACQUIRE) != next_generation) {}
    }
#endif
}

inline void sync_threads(uint32_t barrier_idx = 0) {
    wait_threads(get_num_threads(), barrier_idx);
}
#endif  // !COMPILE_FOR_TRISC

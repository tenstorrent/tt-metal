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

#endif  // !COMPILE_FOR_TRISC

// Semaphores 27–30: thread barriers; 31: watcher ring buffer.
constexpr uintptr_t TENSIX_GLOBAL_SEM_BASE = 0x01840000;
constexpr uint32_t TENSIX_GLOBAL_SEM_STRIDE = 0x40;
constexpr uint32_t COMPUTE_BARRIER_ARRIVED_SEM_IDX = 30;
constexpr uint32_t COMPUTE_BARRIER_GENERATION_SEM_IDX = 29;
constexpr uint32_t ALL_CORES_BARRIER_ARRIVED_SEM_IDX = 28;
constexpr uint32_t ALL_CORES_BARRIER_GENERATION_SEM_IDX = 27;
constexpr uint32_t TENSIX_GLOBAL_SEM_VALUE_MASK = 0xFFFFu;

inline volatile uint32_t* tensix_global_sem(uint32_t idx) {
    return reinterpret_cast<volatile uint32_t*>(TENSIX_GLOBAL_SEM_BASE + idx * TENSIX_GLOBAL_SEM_STRIDE);
}

inline void tensix_global_sem_init(uint32_t idx, uint32_t value) { *tensix_global_sem(idx) = value; }

inline uint32_t tensix_global_sem_read(uint32_t idx) { return *tensix_global_sem(idx); }

// A read at +4*(inc+8) posts `inc` and returns the pre-increment value (same alias as the watcher ring buffer).
inline uint32_t tensix_global_sem_fetch_add(uint32_t idx, uint32_t inc) {
    return *reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uintptr_t>(tensix_global_sem(idx)) + 4 * (inc + 8));
}

#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t kTriscCoresPerNeo = 4;
#endif

#endif  // ARCH_QUASAR

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
    tensix_global_sem_init(COMPUTE_BARRIER_ARRIVED_SEM_IDX, 0);
    tensix_global_sem_init(COMPUTE_BARRIER_GENERATION_SEM_IDX, 0);
    tensix_global_sem_init(ALL_CORES_BARRIER_ARRIVED_SEM_IDX, 0);
    tensix_global_sem_init(ALL_CORES_BARRIER_GENERATION_SEM_IDX, 0);
#endif
}
#endif  // !COMPILE_FOR_TRISC

inline void tensix_global_sem_barrier(uint32_t arrived_idx, uint32_t generation_idx, uint32_t participants) {
#if defined(ARCH_QUASAR)
    if (participants <= 1) {
        return;
    }
    asm volatile("fence rw, rw" ::: "memory");
    uint32_t next_generation = (tensix_global_sem_read(generation_idx) + 1) & TENSIX_GLOBAL_SEM_VALUE_MASK;
    uint32_t arrived = tensix_global_sem_fetch_add(arrived_idx, 1) + 1;
    if (arrived == participants) {
        tensix_global_sem_init(arrived_idx, 0);
        // Arrival reset must be visible before we bump generation, or a waiter can
        // leave, re-enter, and increment a stale count.
        asm volatile("fence w, w" ::: "memory");
        tensix_global_sem_init(generation_idx, next_generation);
    } else {
        while ((tensix_global_sem_read(generation_idx) & TENSIX_GLOBAL_SEM_VALUE_MASK) != next_generation) {
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
#else
    (void)arrived_idx;
    (void)generation_idx;
    (void)participants;
#endif
}

// barrier_idx selects an independent barrier so co-resident kernels with different
// participant counts (e.g. a DFB's producer vs consumer kernel) don't share a counter.
inline void wait_threads(uint32_t participants, uint32_t barrier_idx = 0) {
    if (participants <= 1) {
        return;
    }

#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_TRISC)
    (void)barrier_idx;  // compute has one semaphore pair; barrier_idx is DM-only
    tensix_global_sem_barrier(COMPUTE_BARRIER_ARRIVED_SEM_IDX, COMPUTE_BARRIER_GENERATION_SEM_IDX, participants);
#else
    volatile KernelBarrier& barrier = g_kernel_barrier[barrier_idx];
    uint32_t next_generation = __atomic_load_n(&barrier.generation, __ATOMIC_ACQUIRE) + 1;
    uint32_t arrived = __atomic_add_fetch(&barrier.arrived, 1, __ATOMIC_ACQ_REL);
    if (arrived == participants) {
        __atomic_store_n(&barrier.arrived, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&barrier.generation, next_generation, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&barrier.generation, __ATOMIC_ACQUIRE) != next_generation) {
        }
    }
#endif  // COMPILE_FOR_TRISC
#endif  // ARCH_QUASAR
}

inline void sync_threads(uint32_t barrier_idx = 0) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
    wait_threads(get_num_threads() * kTriscCoresPerNeo, barrier_idx);
#else
    wait_threads(get_num_threads(), barrier_idx);
#endif
}

inline void sync_all_cores(uint32_t participants) {
#if defined(ARCH_QUASAR)
    tensix_global_sem_barrier(ALL_CORES_BARRIER_ARRIVED_SEM_IDX, ALL_CORES_BARRIER_GENERATION_SEM_IDX, participants);
#else
    (void)participants;
#endif
}

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

// Shared barrier state for DM kernels on a worker.
extern volatile KernelBarrier g_kernel_barrier;

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
    g_kernel_barrier.arrived = 0;
    g_kernel_barrier.generation = 0;
#endif
}

inline void wait_threads(uint32_t participants) {
    if (participants <= 1) {
        return;
    }

#if defined(ARCH_QUASAR)
    uint32_t next_generation = __atomic_load_n(&g_kernel_barrier.generation, __ATOMIC_ACQUIRE) + 1;
    uint32_t arrived = __atomic_add_fetch(&g_kernel_barrier.arrived, 1, __ATOMIC_ACQ_REL);
    if (arrived == participants) {
        __atomic_store_n(&g_kernel_barrier.arrived, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_kernel_barrier.generation, next_generation, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&g_kernel_barrier.generation, __ATOMIC_ACQUIRE) != next_generation) {}
    }
#endif
}

inline void sync_threads() {
    wait_threads(get_num_threads());
}
#endif  // !COMPILE_FOR_TRISC

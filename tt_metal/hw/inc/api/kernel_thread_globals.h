// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#if defined(ARCH_QUASAR)
#include "core_config.h"  // MaxDMProcessorsPerCoreType
// The DM + compute barrier expands a tensix count into TRISCs, so it needs NUM_TRISC_CORES.
// Only DM and TRISC builds take part in it (and only they can reach dev_mem_map.h), so eth
// and host builds fall back to the no-op path below.
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM)
#define TT_HAS_DM_COMPUTE_BARRIER 1
#include "dev_mem_map.h"  // NUM_TRISC_CORES
#endif
#endif

#if defined(ARCH_QUASAR)

// ---------------------------------------------------------------------------
// Kernel thread synchronization on Quasar
//
// Three rendezvous flavors live in this header. They differ in who
// participates and, because of that, in which hardware they are built on:
//
//   1. The DM threads of one kernel -- sync_threads() from a DM kernel.
//      Participants are the DM harts running that kernel (get_num_threads()).
//      DM harts share a coherent view of L1, so this is a generation-counting
//      barrier held in L1 (g_kernel_barrier) driven by plain L1 atomics.
//      Each kernel gets its own slot in that array, selected by the barrier id
//      firmware derived for it (get_barrier_id()). Co-resident DM kernels
//      therefore never share a counter, which matters because the release is
//      keyed on the arriving hart's own participant count: two kernels with
//      different thread counts sharing one counter would never agree on a
//      target and would hang.
//
//   2. The TRISCs of one compute kernel -- sync_threads() from a compute
//      kernel. Participants are every TRISC of every NEO the kernel occupies,
//      i.e. get_num_threads() * NUM_TRISC_CORES, so the barrier releases only
//      once unpack, math, pack and SFPU on all of those NEOs have arrived.
//      The L1 barrier of flavor 1 cannot be reused: the TRISC roles do not
//      share a coherent view of L1, so an arrival written by one role is not
//      guaranteed to be seen by another. This rides on a pair of tensix global
//      semaphores instead, which are registers every TRISC reads identically.
//
//   3. DM threads and TRISCs together -- dm_compute_barrier(). Participants
//      span both processor kinds, so neither the DM-only L1 barrier nor the
//      compute pair alone will do. It uses a second, independent tensix global
//      semaphore pair, which keeps a mixed rendezvous from corrupting the
//      count of a compute-only sync_threads() that is in flight, and vice
//      versa. Participants are counted in the units a user parallelizes over,
//      DM harts and tensix engines, and the expansion to individual TRISCs
//      happens in here. The count is passed in because no single processor
//      knows how many other kernels are co-resident on the core.
//
// All of the above are intra-core: they synchronize processors on one core and
// say nothing about other cores. Cross-core synchronization is a NOC semaphore
// (see noc_semaphore.h), not one of these barriers.
// ---------------------------------------------------------------------------

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

// One barrier slot per DM hart, which is the most co-resident DM kernels a core can host
// (the degenerate case being every hart running its own single-threaded kernel). Slots are
// indexed by a kernel's lowest hart, so no two co-resident kernels can share a counter
// regardless of topology.
constexpr uint32_t NUM_KERNEL_BARRIERS = MaxDMProcessorsPerCoreType;
extern volatile KernelBarrier g_kernel_barrier[NUM_KERNEL_BARRIERS];

// Barrier slot for the kernel on this hart, set by dmk.cc before the kernel runs.
// Read through get_barrier_id().
extern thread_local uint32_t my_barrier_id;

#endif  // !COMPILE_FOR_TRISC

// Semaphores 27–30: thread barriers; 31: watcher ring buffer.
constexpr uintptr_t TENSIX_GLOBAL_SEM_BASE = 0x01840000;
constexpr uint32_t TENSIX_GLOBAL_SEM_STRIDE = 0x40;
constexpr uint32_t COMPUTE_BARRIER_ARRIVED_SEM_IDX = 30;
constexpr uint32_t COMPUTE_BARRIER_GENERATION_SEM_IDX = 29;
constexpr uint32_t DM_COMPUTE_BARRIER_ARRIVED_SEM_IDX = 28;
constexpr uint32_t DM_COMPUTE_BARRIER_GENERATION_SEM_IDX = 27;
constexpr uint32_t TENSIX_GLOBAL_SEM_VALUE_MASK = 0xFFFFu;

inline volatile uint32_t* tensix_global_sem(uint32_t idx) {
    return reinterpret_cast<volatile uint32_t*>(TENSIX_GLOBAL_SEM_BASE + idx * TENSIX_GLOBAL_SEM_STRIDE);
}

inline void tensix_global_sem_init(uint32_t idx, uint32_t value) { *tensix_global_sem(idx) = value; }

inline uint32_t tensix_global_sem_read(uint32_t idx) { return *tensix_global_sem(idx); }

// A read at +4*(inc+8) posts `inc` and returns the pre-increment value (same alias as the watcher ring buffer).
// Only inc=1 is exercised; the upper bound the alias supports is not documented in tensix_neo_reg.h.
inline uint32_t tensix_global_sem_fetch_add(uint32_t idx, uint32_t inc) {
    return *reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uintptr_t>(tensix_global_sem(idx)) + 4 * (inc + 8));
}

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
    tensix_global_sem_init(DM_COMPUTE_BARRIER_ARRIVED_SEM_IDX, 0);
    tensix_global_sem_init(DM_COMPUTE_BARRIER_GENERATION_SEM_IDX, 0);
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

// clang-format off
/**
 * Returns the barrier slot owned by the calling kernel, derived by firmware so that
 * co-resident kernels rendezvous independently. sync_threads() uses this internally, so
 * kernels do not normally need it; it is exposed for code building its own barrier on top
 * of wait_threads().
 *
 * Compute kernels rendezvous on tensix semaphores rather than an L1 slot, so this returns 0
 * there, as it does on WH/BH.
 *
 * Return value: Barrier slot index, 0 to NUM_KERNEL_BARRIERS - 1.
 */
// clang-format on
inline uint32_t get_barrier_id() {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    return my_barrier_id;
#else
    return 0;
#endif
}

// Rendezvous of `participants` processors, reusable across rounds.
//
// From a DM kernel this is the L1 generation barrier (flavor 1 above), on the slot this
// kernel owns, so co-resident kernels with different participant counts (e.g. a DFB's
// producer vs consumer kernel) cannot share a counter.
//
// From a compute kernel this is the tensix global semaphore pair (flavor 2). There is one
// such pair, and the TRISC roles of a kernel always rendezvous as one group, so there is no
// second group to keep separate and no slot to select.
inline void wait_threads(uint32_t participants) {
    if (participants <= 1) {
        return;
    }

#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_TRISC)
    tensix_global_sem_barrier(COMPUTE_BARRIER_ARRIVED_SEM_IDX, COMPUTE_BARRIER_GENERATION_SEM_IDX, participants);
#else
    volatile KernelBarrier& barrier = g_kernel_barrier[get_barrier_id()];
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

// clang-format off
/**
 * Barrier across the threads of the calling kernel. Reusable, so every thread of the kernel
 * must call it the same number of times and in the same order.
 *
 * From a DM kernel the participants are that kernel's DM harts. From a compute kernel they are
 * all NUM_TRISC_CORES TRISCs of each NEO the kernel occupies, so unpack, math, pack and SFPU
 * are released together. No-op on WH/BH, where a kernel is single-threaded.
 *
 * Which barrier a DM kernel rendezvouses on is decided by firmware, not the caller, so two
 * co-resident kernels cannot collide. This does not synchronize with the other kernels on
 * the core; use dm_compute_barrier() for a rendezvous that spans DM and compute.
 */
// clang-format on
inline void sync_threads() {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
    wait_threads(get_num_threads() * NUM_TRISC_CORES);
#else
    wait_threads(get_num_threads());
#endif
}

// clang-format off
/**
 * Barrier spanning DM threads and compute TRISCs on this core (flavor 3 above), for handing
 * off between a DM kernel and a compute kernel without a DFB or a NOC semaphore. Every
 * participating processor, DM hart and TRISC alike, must call this the same number of times
 * and with the same counts.
 *
 * Counts are given in the units a user parallelizes over: DM harts, and whole tensix engines
 * rather than individual TRISCs. All NUM_TRISC_CORES TRISCs of each engine arrive, and this
 * expands the count for you.
 *
 * Independent of sync_threads(), so a kernel may use both. Intra-core only: it does not
 * synchronize with any other core or node, despite spanning both processor kinds. No-op on
 * WH/BH.
 *
 * | Argument     | Description                                      | Type     | Required |
 * |--------------|--------------------------------------------------|----------|----------|
 * | dm_threads   | Number of DM harts that will arrive.             | uint32_t | True     |
 * | tensixes     | Number of tensix engines (NEOs) that will arrive.| uint32_t | True     |
 */
// clang-format on
inline void dm_compute_barrier(uint32_t dm_threads, uint32_t tensixes) {
#if defined(TT_HAS_DM_COMPUTE_BARRIER)
    tensix_global_sem_barrier(
        DM_COMPUTE_BARRIER_ARRIVED_SEM_IDX,
        DM_COMPUTE_BARRIER_GENERATION_SEM_IDX,
        dm_threads + tensixes * NUM_TRISC_CORES);
#else
    (void)dm_threads;
    (void)tensixes;
#endif
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"  // get_semaphore
#include "dev_mem_map.h"
#include "internal/risc_attribs.h"

namespace sem_internal {

/**
 * @brief One entry in kernel_includes.hpp's list of cached semaphores: which semaphore, and
 *        how many harts on this core use it.
 */
struct CachedSemaphore {
    std::uint32_t id;
    std::uint32_t binder_harts;
};

// Cached-pool entry/exit for the DM_LOCAL_CACHED semaphores a kernel binds. The generated
// kernel_includes.hpp lists them (sem_internal::kCachedSemaphores) and dmk.cc calls these
// around kernel_main(). A cached semaphore's pool row must be seeded with its init value once per
// program, by exactly one hart, before anyone touches it, and is left clean for the next
// program. Each 8B row is [0] = the counter, [1] = a bookkeeping word: entered[15:0],
// exited[30:16], seeded[31]. On entry, each binder hart increments `entered`; whoever got
// there first copies the init value from the ring into the pool counter and sets `seeded`;
// everyone else waits for that bit. On exit, each hart increments `exited`; the last one
// zeroes the bookkeeping word so the next program starts fresh. Any number of local
// kernels/threads works.
__attribute__((always_inline)) inline std::uint32_t* cached_pool_row(std::uint32_t id) {
    return reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(MEM_SEM_CACHED_POOL_BASE) + id * MEM_SEM_CACHED_POOL_ROW);
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void seed_cached_row(const CachedSemaphore& sem) {
    auto* row = cached_pool_row(sem.id);
    if ((__atomic_fetch_add(row + 1, 1u, __ATOMIC_ACQ_REL) & 0xFFFFu) == 0u) {
        row[0] = *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            ::get_semaphore<core_type>(sem.id) + MEM_L1_UNCACHED_BASE);
        __atomic_fetch_or(row + 1, 0x80000000u, __ATOMIC_RELEASE);
    } else {
        while ((__atomic_load_n(row + 1, __ATOMIC_ACQUIRE) & 0x80000000u) == 0u) {
        }
    }
}

__attribute__((always_inline)) inline void restore_cached_row(const CachedSemaphore& sem) {
    auto* row = cached_pool_row(sem.id);
    if (((__atomic_fetch_add(row + 1, 0x10000u, __ATOMIC_ACQ_REL) >> 16) & 0x7FFFu) == sem.binder_harts - 1u) {
        __atomic_store_n(row + 1, 0u, __ATOMIC_RELEASE);
    }
}

// Handles the semaphores one after another with no loop: the recursion over I unrolls at compile
// time into one block per semaphore.
template <ProgrammableCoreType core_type, std::size_t N, std::size_t I = 0>
__attribute__((always_inline)) inline void init_dm_local_cached(const std::array<CachedSemaphore, N>& sems) {
    if constexpr (I < N) {
        seed_cached_row<core_type>(sems[I]);
        init_dm_local_cached<core_type, N, I + 1>(sems);
    }
}

template <std::size_t N, std::size_t I = 0>
__attribute__((always_inline)) inline void finish_dm_local_cached(const std::array<CachedSemaphore, N>& sems) {
    if constexpr (I < N) {
        restore_cached_row(sems[I]);
        finish_dm_local_cached<N, I + 1>(sems);
    }
}

}  // namespace sem_internal

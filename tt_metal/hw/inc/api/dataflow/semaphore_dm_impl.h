// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "api/dataflow/semaphore_binding_token.h"  // SemScope + semaphore_detail::always_false
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "dev_mem_map.h"
#include "tools/profiler/kernel_profiler.hpp"  // SYNC_SIGNAL / SYNC_WAIT

namespace semaphore_detail {

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline std::uintptr_t sem_l1_offset(std::uint32_t id, SemScope scope) {
    // COMPUTE_ATOMIC is the Blackhole Tensix hardware semaphore, which a DM core cannot reach; the host
    // rejects such a binding (ValidateProgramSpec). Kept as an assert so a relaxed host rule can never
    // silently take the non-atomic path.
    ASSERT(scope != SemScope::COMPUTE_ATOMIC);  // COMPUTE_ATOMIC semaphores may not be bound by a DM kernel
#ifdef ARCH_QUASAR
    if (scope == SemScope::DM_LOCAL_CACHED) {
        ASSERT(id < MEM_SEM_CACHED_POOL_SIZE / MEM_SEM_CACHED_POOL_ROW);
        return static_cast<std::uintptr_t>(MEM_SEM_CACHED_POOL_BASE) + id * MEM_SEM_CACHED_POOL_ROW;
    }
#endif
    return get_semaphore<core_type>(id);
}

__attribute__((always_inline)) inline volatile tt_l1_ptr std::uint32_t* local_ptr(
    std::uintptr_t l1_offset, SemScope scope) {
    std::uintptr_t addr = l1_offset;
#ifdef ARCH_QUASAR
    if (scope != SemScope::DM_LOCAL_CACHED) {
        addr += MEM_L1_UNCACHED_BASE;
    }
#endif
    return reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

__attribute__((always_inline)) inline std::uint32_t load(std::uintptr_t l1_offset, SemScope scope) {
#ifdef ARCH_QUASAR
    if (scope == SemScope::DM_LOCAL_CACHED) {
        return __atomic_load_n(reinterpret_cast<std::uint32_t*>(l1_offset), __ATOMIC_RELAXED);
    }
#endif
    invalidate_l1_cache();
    return *local_ptr(l1_offset, scope);
}

// Settled read. On DM this is just load(): every write this core makes to a semaphore word is a
// RISC-side atomic or store that has retired before the next instruction runs, so there is nothing
// of ours in flight to fence. The compute implementation needs a real fence here because its atomic
// is posted into the Tensix pipe -- see semaphore_compute_impl.h::current().
template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline std::uint32_t current(std::uintptr_t l1_offset, SemScope scope) {
    return load(l1_offset, scope);
}

#if defined(ARCH_QUASAR) && !defined(TT_EMULE_USE_L1_POOL)
template <ProgrammableCoreType core_type>
inline std::uint32_t external_lock_l1_offset(std::uintptr_t l1_offset) {
    const std::uint32_t id =
        (static_cast<std::uint32_t>(l1_offset) - static_cast<std::uint32_t>(get_semaphore<core_type>(0))) /
        L1_ALIGNMENT;
    ASSERT(id * L1_ALIGNMENT < MEM_SEM_LOCK_SIZE);
    return MEM_SEM_LOCK_BASE + id * L1_ALIGNMENT;
}

inline std::uint32_t cas_ret_slot() {
    std::uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    ASSERT(static_cast<std::uint32_t>(hart) * 4 < MEM_SEM_CAS_RET_SIZE);
    return MEM_SEM_CAS_RET_BASE + static_cast<std::uint32_t>(hart) * 4;
}
#endif

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void up(std::uintptr_t l1_offset, SemScope scope, std::uint32_t value) {
    if (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        SYNC_SIGNAL("SYNC-SEM-SET", l1_offset);
        __atomic_add_fetch(reinterpret_cast<std::uint32_t*>(l1_offset), value, __ATOMIC_SEQ_CST);
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
    } else if (scope == SemScope::EXTERNAL) {
        noc_semaphore_inc(::get_noc_addr(l1_offset), value);
        noc_async_atomic_barrier();
    } else {
        SYNC_SIGNAL("SYNC-SEM-SET", l1_offset);
        *local_ptr(l1_offset, scope) += value;
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void down(std::uintptr_t l1_offset, SemScope scope, std::uint32_t value) {
    auto* sem_addr = local_ptr(l1_offset, scope);
    WAYPOINT("NSDW");
    if (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        auto* word = reinterpret_cast<std::uint32_t*>(l1_offset);
        std::uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
        {
            SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
            do {
                while (observed < value) {
                    observed = __atomic_load_n(word, __ATOMIC_RELAXED);
                }
                WAYPOINT("NSDD");
            } while (!__atomic_compare_exchange_n(
                word, &observed, observed - value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        }
        SYNC_SIGNAL("SYNC-SEM-SET", l1_offset);
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
    } else if (scope == SemScope::EXTERNAL) {
#if defined(ARCH_QUASAR) && !defined(TT_EMULE_USE_L1_POOL) && !defined(NOC_API_V1)
        noc_async_atomic_barrier();
        const std::uint64_t sem_noc = ::get_noc_addr(l1_offset);
        const std::uint64_t lock_noc = ::get_noc_addr(external_lock_l1_offset<core_type>(l1_offset));
        const std::uint32_t ret_slot = cas_ret_slot();
        auto* ret_word = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            static_cast<std::uintptr_t>(MEM_L1_UNCACHED_BASE) + ret_slot);
        const auto lock_cas = [&](std::uint32_t cmp4, std::uint32_t swap4) -> std::uint32_t {
            noc_fast_atomic_cas4<DM_DEDICATED_NOC>(noc_index, lock_noc, NOC_UNICAST_WRITE_VC, cmp4, swap4, ret_slot);
            noc_async_atomic_barrier();
            return *ret_word;
        };
        for (;;) {
            {
                SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
                do {
                    invalidate_l1_cache();
                } while (*sem_addr < value);
            }
            if (lock_cas(0, 1) != 0) {
                continue;
            }
            invalidate_l1_cache();
            const bool sufficient = *sem_addr >= value;
            if (sufficient) {
                WAYPOINT("NSDD");
                noc_semaphore_inc(sem_noc, static_cast<std::uint32_t>(0u - value));
                noc_async_atomic_barrier();
            }
            lock_cas(1, 0);
            if (sufficient) {
                noc_restore_default_atomic_ret_addr(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                return;
            }
        }
#else
        {
            SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
            do {
                invalidate_l1_cache();
            } while (*sem_addr < value);
        }
        WAYPOINT("NSDD");
        noc_semaphore_inc(::get_noc_addr(l1_offset), static_cast<std::uint32_t>(0u - value));
        noc_async_atomic_barrier();
#endif
    } else {
        {
            SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
            do {
                invalidate_l1_cache();
            } while (*sem_addr < value);
        }
        WAYPOINT("NSDD");
        SYNC_SIGNAL("SYNC-SEM-SET", l1_offset);
        *sem_addr -= value;
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void wait(std::uintptr_t l1_offset, SemScope scope, std::uint32_t value) {
    if (scope == SemScope::DM_LOCAL_CACHED) {
        WAYPOINT("NSW");
        {
            SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
            while (load(l1_offset, scope) != value) {
            }
        }
        WAYPOINT("NSD");
    } else {
        noc_semaphore_wait(local_ptr(l1_offset, scope), value);
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void wait_min(std::uintptr_t l1_offset, SemScope scope, std::uint32_t value) {
    if (scope == SemScope::DM_LOCAL_CACHED) {
        WAYPOINT("NSMW");
        {
            SYNC_WAIT("SYNC-SEM-WAIT", l1_offset);
            while (load(l1_offset, scope) < value) {
            }
        }
        WAYPOINT("NSMD");
    } else {
        noc_semaphore_wait_min(local_ptr(l1_offset, scope), value);
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void set(std::uintptr_t l1_offset, SemScope scope, std::uint32_t value) {
    noc_semaphore_set(local_ptr(l1_offset, scope), value);
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void up_remote(
    std::uintptr_t l1_offset,
    SemScope scope,
    const Noc& noc,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t value,
    std::uint8_t vc) {
    if (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        ASSERT(noc.is_local_bank(noc_x, noc_y));
        up<core_type>(l1_offset, scope, value);
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
        return;
    }
    const std::uint64_t dest_noc_addr = ::get_noc_addr(noc_x, noc_y, l1_offset, noc.get_noc_id());
    noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
}

inline void relay_unicast(
    std::uintptr_t src_l1_offset,
    std::uintptr_t dst_l1_offset,
    const Noc& noc,
    std::uint32_t noc_x,
    std::uint32_t noc_y) {
    ASSERT(src_l1_offset != dst_l1_offset);
    const std::uint64_t dst_noc_addr = ::get_noc_addr(noc_x, noc_y, dst_l1_offset, noc.get_noc_id());
    noc_semaphore_set_remote(src_l1_offset, dst_noc_addr, noc.get_noc_id());
}

template <NocOptions opts>
inline void set_multicast(
    std::uintptr_t l1_offset,
    const Noc& noc,
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t num_dests,
    bool linked) {
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, l1_offset, noc.get_noc_id());
    if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
        noc_semaphore_set_multicast_loopback_src(l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    } else {
        noc_semaphore_set_multicast(l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    }
}

template <NocOptions opts>
inline void relay_multicast(
    std::uintptr_t src_l1_offset,
    std::uintptr_t dst_l1_offset,
    const Noc& noc,
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t num_dests,
    bool linked) {
    ASSERT(src_l1_offset != dst_l1_offset);
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, dst_l1_offset, noc.get_noc_id());
    if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
        noc_semaphore_set_multicast_loopback_src(src_l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    } else {
        noc_semaphore_set_multicast(src_l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    }
}

inline void inc_multicast(
    std::uintptr_t l1_offset,
    const Noc& noc,
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t value,
    std::uint32_t num_dests) {
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, l1_offset, noc.get_noc_id());
    noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
}

}  // namespace semaphore_detail

#ifdef ARCH_QUASAR
namespace sem_internal {

// Cached-pool entry/exit for the DM_LOCAL_CACHED semaphores a kernel binds. The generated
// header lists them (sem_internal::kCachedSemaphores) and these are called around
// kernel_main(). A cached semaphore's pool row must be seeded with its init value once per
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
__attribute__((always_inline)) inline void init_dm_local_cached(const CachedSemaphore (&sems)[N]) {
    if constexpr (I < N) {
        seed_cached_row<core_type>(sems[I]);
        init_dm_local_cached<core_type, N, I + 1>(sems);
    }
}

template <std::size_t N, std::size_t I = 0>
__attribute__((always_inline)) inline void finish_dm_local_cached(const CachedSemaphore (&sems)[N]) {
    if constexpr (I < N) {
        restore_cached_row(sems[I]);
        finish_dm_local_cached<N, I + 1>(sems);
    }
}

}  // namespace sem_internal
#endif

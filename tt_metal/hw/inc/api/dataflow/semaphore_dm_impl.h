// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "dev_mem_map.h"

namespace semaphore_detail {

// Dependent false, so a static_assert in a class-template member fires only on instantiation.
template <SemScope>
inline constexpr bool always_false = false;

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline std::uintptr_t sem_l1_offset(std::uint32_t id) {
    // COMPUTE_ATOMIC is the Blackhole Tensix hardware semaphore, which a DM core cannot reach; the host
    // rejects such a binding (ValidateProgramSpec). Kept as a build failure so a relaxed host rule can
    // never silently take the non-atomic path.
    static_assert(scope != SemScope::COMPUTE_ATOMIC, "COMPUTE_ATOMIC semaphores may not be bound by a DM kernel");
#ifdef ARCH_QUASAR
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
        ASSERT(id < MEM_DM_CACHED_SEM_SIZE / MEM_DM_CACHED_SEM_ROW);
        return static_cast<std::uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * MEM_DM_CACHED_SEM_ROW;
    }
#endif
    return get_semaphore<core_type>(id);
}

template <SemScope scope>
__attribute__((always_inline)) inline volatile tt_l1_ptr std::uint32_t* local_ptr(std::uintptr_t l1_offset) {
    std::uintptr_t addr = l1_offset;
#ifdef ARCH_QUASAR
    if constexpr (scope != SemScope::DM_LOCAL_CACHED) {
        addr += MEM_L1_UNCACHED_BASE;
    }
#endif
    return reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t load(std::uintptr_t l1_offset) {
#ifdef ARCH_QUASAR
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
        return __atomic_load_n(reinterpret_cast<std::uint32_t*>(l1_offset), __ATOMIC_RELAXED);
    }
#endif
    invalidate_l1_cache();
    return *local_ptr<scope>(l1_offset);
}

// Settled read. On DM this is just load(): every write this core makes to a semaphore word is a
// RISC-side atomic or store that has retired before the next instruction runs, so there is nothing
// of ours in flight to fence. The compute implementation needs a real fence here because its atomic
// is posted into the Tensix pipe -- see semaphore_compute_impl.h::current().
template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t current(std::uintptr_t l1_offset) {
    return load<scope>(l1_offset);
}

#if defined(ARCH_QUASAR) && !defined(TT_EMULE_USE_L1_POOL)
template <ProgrammableCoreType core_type>
inline std::uint32_t external_lock_l1_offset(std::uintptr_t l1_offset) {
    const std::uint32_t id =
        (static_cast<std::uint32_t>(l1_offset) - static_cast<std::uint32_t>(get_semaphore<core_type>(0))) /
        L1_ALIGNMENT;
    ASSERT(id * L1_ALIGNMENT < MEM_NOC_SEM_LOCK_SIZE);
    return MEM_NOC_SEM_LOCK_BASE + id * L1_ALIGNMENT;
}

inline std::uint32_t cas_ret_slot() {
    std::uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    ASSERT(static_cast<std::uint32_t>(hart) * 4 < MEM_NOC_CAS_RET_SIZE);
    return MEM_NOC_CAS_RET_BASE + static_cast<std::uint32_t>(hart) * 4;
}
#endif

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void up(std::uintptr_t l1_offset, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        __atomic_add_fetch(reinterpret_cast<std::uint32_t*>(l1_offset), value, __ATOMIC_SEQ_CST);
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
    } else if constexpr (scope == SemScope::EXTERNAL) {
        noc_semaphore_inc(::get_noc_addr(l1_offset), value);
        noc_async_atomic_barrier();
    } else {
        *local_ptr<scope>(l1_offset) += value;
    }
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void down(std::uintptr_t l1_offset, std::uint32_t value) {
    auto* sem_addr = local_ptr<scope>(l1_offset);
    WAYPOINT("NSDW");
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        auto* word = reinterpret_cast<std::uint32_t*>(l1_offset);
        std::uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
        do {
            while (observed < value) {
                observed = __atomic_load_n(word, __ATOMIC_RELAXED);
            }
            WAYPOINT("NSDD");
        } while (
            !__atomic_compare_exchange_n(word, &observed, observed - value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
    } else if constexpr (scope == SemScope::EXTERNAL) {
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
            do {
                invalidate_l1_cache();
            } while (*sem_addr < value);
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
        do {
            invalidate_l1_cache();
        } while (*sem_addr < value);
        WAYPOINT("NSDD");
        noc_semaphore_inc(::get_noc_addr(l1_offset), static_cast<std::uint32_t>(0u - value));
        noc_async_atomic_barrier();
#endif
    } else {
        do {
            invalidate_l1_cache();
        } while (*sem_addr < value);
        WAYPOINT("NSDD");
        *sem_addr -= value;
    }
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait(std::uintptr_t l1_offset, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
        WAYPOINT("NSW");
        while (load<scope>(l1_offset) != value) {
        }
        WAYPOINT("NSD");
    } else {
        noc_semaphore_wait(local_ptr<scope>(l1_offset), value);
    }
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait_min(std::uintptr_t l1_offset, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
        WAYPOINT("NSMW");
        while (load<scope>(l1_offset) < value) {
        }
        WAYPOINT("NSMD");
    } else {
        noc_semaphore_wait_min(local_ptr<scope>(l1_offset), value);
    }
}

template <SemScope scope>
__attribute__((always_inline)) inline void set(std::uintptr_t l1_offset, std::uint32_t value) {
    noc_semaphore_set(local_ptr<scope>(l1_offset), value);
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void up_remote(
    std::uintptr_t l1_offset,
    const Noc& noc,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t value,
    std::uint8_t vc) {
    if constexpr (scope == SemScope::DM_LOCAL_CACHED) {
#ifdef ARCH_QUASAR
        ASSERT(noc.is_local_bank(noc_x, noc_y));
        up<core_type, scope>(l1_offset, value);
#else
        ASSERT(false);  // the host census never bakes DM_LOCAL_CACHED for this platform
#endif
        return;
    }
    const std::uint64_t dest_noc_addr = ::get_noc_addr(noc_x, noc_y, l1_offset, noc.get_noc_id());
    noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
}

template <SemScope src_scope, SemScope dst_scope>
inline void relay_unicast(
    std::uintptr_t src_l1_offset,
    std::uintptr_t dst_l1_offset,
    const Noc& noc,
    std::uint32_t noc_x,
    std::uint32_t noc_y) {
    static_assert(src_scope != SemScope::DM_LOCAL_CACHED, "relay is unavailable on a cached semaphore");
    static_assert(dst_scope != SemScope::DM_LOCAL_CACHED, "relay cannot target a cached semaphore");
    ASSERT(src_l1_offset != dst_l1_offset);
    const std::uint64_t dst_noc_addr = ::get_noc_addr(noc_x, noc_y, dst_l1_offset, noc.get_noc_id());
    noc_semaphore_set_remote(src_l1_offset, dst_noc_addr, noc.get_noc_id());
}

template <NocOptions opts, SemScope scope>
inline void set_multicast(
    std::uintptr_t l1_offset,
    const Noc& noc,
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t num_dests,
    bool linked) {
    static_assert(scope != SemScope::DM_LOCAL_CACHED, "multicast is unavailable on a cached semaphore");
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, l1_offset, noc.get_noc_id());
    if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
        noc_semaphore_set_multicast_loopback_src(l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    } else {
        noc_semaphore_set_multicast(l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    }
}

template <NocOptions opts, SemScope src_scope, SemScope dst_scope>
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
    static_assert(src_scope != SemScope::DM_LOCAL_CACHED, "relay is unavailable on a cached semaphore");
    static_assert(dst_scope != SemScope::DM_LOCAL_CACHED, "relay cannot target a cached semaphore");
    ASSERT(src_l1_offset != dst_l1_offset);
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, dst_l1_offset, noc.get_noc_id());
    if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
        noc_semaphore_set_multicast_loopback_src(src_l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    } else {
        noc_semaphore_set_multicast(src_l1_offset, multicast_addr, num_dests, linked, noc.get_noc_id());
    }
}

template <SemScope scope>
inline void inc_multicast(
    std::uintptr_t l1_offset,
    const Noc& noc,
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t value,
    std::uint32_t num_dests) {
    static_assert(scope != SemScope::DM_LOCAL_CACHED, "multicast is unavailable on a cached semaphore");
    const std::uint64_t multicast_addr =
        ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, l1_offset, noc.get_noc_id());
    noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
}

}  // namespace semaphore_detail

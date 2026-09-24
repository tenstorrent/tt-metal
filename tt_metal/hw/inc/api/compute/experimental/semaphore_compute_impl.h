// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/dataflow/semaphore_binding_token.h"  // SemScope + semaphore_detail::always_false
#include "core_config.h"
#include "ckernel.h"

// Blackhole compute semaphore (SemScope::COMPUTE_ATOMIC): UNPACK <-> PACK synchronization on the Tensix
// hardware (Sync Unit) semaphore UNPACK_OPERAND_SYNC, index 3. Bound only by compute kernels; used by
// TRISC0 (UNPACK) and TRISC2 (PACK). Every operation is a Tensix instruction gated by this thread's Wait
// Gate, never a RISC read-modify-write.
//
// Primitives (Value and Max are 4-bit fields of the hardware semaphore):
//   up(n)           n x [ STALLWAIT(my engine idle) ; SEMPOST ]     RISC returns; credit lands after the
//                                                                    packer/unpacker work (publish-after-data)
//   down(n)         n x [ STALLWAIT(my engine idle) ; SEMGET  ]     RISC returns; release after the engine read
//   wait_min(1)     SEMWAIT(block my engine while Value == 0)       RISC returns; holds UNPACR/PACR
//   wait_not_full() SEMWAIT(block my engine while Value >= Max)     RISC returns; producer back-pressure
//   wait_not_full(n>1) tensix_sync ; RISC polls Value <= Max - n    RISC blocks; room for a batched up(n)
//   wait(v)         tensix_sync ; RISC polls Value == v              RISC blocks
//   wait_min(v!=1)  tensix_sync ; RISC polls Value >= v              RISC blocks
//   value()         tensix_sync ; read Value                         RISC blocks; settled value
//   set(v)          STALLWAIT(my engine idle) ; SEMINIT(Max, v)      RISC returns
// The tensix_sync in the RISC-poll forms retires this thread's own queued SEMPOST/SEMGET first; without
// it the read can predate them. SEMWAIT has only the "== 0" and ">= Max" conditions.
//
// Input / output: Max = COMPUTE_SEMAPHORE_MAX (host-baked from SemaphoreAdvancedOptions::max_value,
// default 15) programmed by every SEMINIT; Value starts at 0, seeded by compute_kernel_hw_startup() on
// PACK, and must be left at 0 by the kernel. Canonical ring use, capacity = ring depth:
//   PACK:   wait_not_full(); pack_tile(ring, slot); up(1);
//   UNPACK: wait_min(1); copy_tile(ring, slot); down(1);
//
// Limits: Value 0..15, SEMPOST saturates and SEMGET floors silently; core-local (no NoC, no DM core);
// index 3 is the only free hardware semaphore, so one compute semaphore per program (host-enforced);
// waits order this thread's engine instructions, not RISC L1 loads (use value() to gate a RISC read).
// Design rationale and measurements: PR #56189.
//
// Blackhole DM <-> compute semaphore (SemScope::DM_COMPUTE_ATOMICS): the host-allocated L1 semaphore word
// (sem_l1_base[TENSIX] + id * L1_ALIGNMENT, the word DM kernels see), bound by both DM and compute kernels.
// Every update is an L1-bank atomic: ThCon ATINCGET here, NoC atomics on DM (semaphore_dm_impl.h), so they
// interleave without losing an update. 32-bit, host-initialized from initial_value, no Sync Unit index used.
//   up(n) / down(n)  STALLWAIT(ThCon + my engine idle) ; ATINCGET(+n / -n)   RISC returns (publish-after-data)
//   wait / wait_min  tensix_sync ; fenced RISC poll of the L1 word          RISC blocks
//   value()          tensix_sync ; fenced RISC load                          RISC blocks
//   set(v)           STALLWAIT(ThCon + my engine idle) ; 32-bit STOREIND      RISC returns; plain store
//   wait_not_full    not available (no hardware Max)
// down() does not wait for sufficiency, as for COMPUTE_ATOMIC: `wait_min(n); <engine reads slot>; down(n)`.
// One decrementer per semaphore (BH has no 32-bit conditional atomic decrement).

// The compute semaphore's Max: host-baked from SemaphoreAdvancedOptions::max_value when set, else the
// hardware ceiling. Every SEMINIT in this file programs it.
#ifndef COMPUTE_SEMAPHORE_MAX
#define COMPUTE_SEMAPHORE_MAX 15
#endif
inline constexpr std::uint32_t kComputeSemaphoreMax = COMPUTE_SEMAPHORE_MAX;
static_assert(kComputeSemaphoreMax >= 1 && kComputeSemaphoreMax <= 15, "COMPUTE_SEMAPHORE_MAX must be 1..15");

/**
 * @brief Seed the compute semaphore for this kernel. Called by compute_kernel_hw_startup() (2.0) on the
 * PACK thread; kernels do not call it directly.
 *
 * Sets the Tensix hardware semaphore backing SemScope::COMPUTE_ATOMIC to Value 0 and Max =
 * COMPUTE_SEMAPHORE_MAX. Must run on the producing thread (PACK), whose first up() is then queued behind
 * it; requires that the previous kernel left the semaphore balanced at 0, which is why the host rejects
 * a nonzero initial_value on a compute binding. No-op on non-Blackhole builds and on the other threads.
 */
__attribute__((always_inline)) inline void compute_semaphore_hw_startup() {
#if defined(ARCH_BLACKHOLE) && defined(TRISC_PACK)
    ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_OPERAND_SYNC, /*value=*/0, kComputeSemaphoreMax);
#endif
}

namespace semaphore_detail {

// COMPUTE_ATOMIC: the bound id selects nothing: every such semaphore is the one free Sync Unit index. The
// function keeps the name sem_l1_offset so the shared Semaphore class template (semaphore.h) needs no
// change; the returned value is the hardware index, threaded through up/down/wait/... below.
// DM_COMPUTE_ATOMICS: the L1 word's byte address.
template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline std::uintptr_t sem_l1_offset(std::uint32_t id) {
    static_assert(core_type == ProgrammableCoreType::TENSIX, "compute semaphores require a Tensix core");
    static_assert(
        scope == SemScope::COMPUTE_ATOMIC || scope == SemScope::DM_COMPUTE_ATOMICS,
        "Blackhole compute supports COMPUTE_ATOMIC and DM_COMPUTE_ATOMICS only");
#if defined(ARCH_BLACKHOLE)
    if constexpr (scope == SemScope::DM_COMPUTE_ATOMICS) {
        return reinterpret_cast<std::uintptr_t>(sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)]) +
               id * L1_ALIGNMENT;
    }
    return ckernel::semaphore::UNPACK_OPERAND_SYNC;
#else
    // Other archs' LLKs do not define this index; fail only if a kernel actually constructs one.
    static_assert(always_false<scope>, "compute semaphores are Blackhole-only");
    return 0;
#endif
}

#if defined(ARCH_BLACKHOLE) && (defined(TRISC_UNPACK) || defined(TRISC_PACK))

// STALLWAIT condition naming this thread's own engine (so up()/down() are ordered after its work), and
// the matching SEMWAIT block bit (so a wait holds exactly this thread's engine instructions).
#if defined(TRISC_UNPACK)
inline constexpr std::uint32_t kEngineIdle = ckernel::p_stall::UNPACK;         // C1|C2: both unpackers idle
inline constexpr std::uint32_t kEngineBlock = ckernel::p_stall::STALL_UNPACK;  // B3: block UNPACR
using SemGpr = ckernel::p_gpr_unpack;
#else
inline constexpr std::uint32_t kEngineIdle = ckernel::p_stall::PACK;         // C3: packer idle
inline constexpr std::uint32_t kEngineBlock = ckernel::p_stall::STALL_PACK;  // B2: block PACR
using SemGpr = ckernel::p_gpr_pack;
#endif

// DM_COMPUTE_ATOMICS L1 word: fresh read (the RISC L0 data cache is not coherent with ATINCGET / NoC writes).
__attribute__((always_inline)) inline std::uint32_t l1_word_load(std::uintptr_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

// DM_COMPUTE_ATOMICS: add `delta` mod 2^32 with ATINCGET, ordered after this thread's engine work.
__attribute__((always_inline)) inline void l1_word_atomic_add(std::uintptr_t addr, std::uint32_t delta) {
    ckernel::t6_atomic_add_l1<SemGpr::SEM_ADDR, SemGpr::SEM_DATA, kEngineIdle>(addr, delta);
}

template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t load(std::uintptr_t handle) {
    if constexpr (scope == SemScope::DM_COMPUTE_ATOMICS) {
        return l1_word_load(handle);
    } else {
        static_assert(scope == SemScope::COMPUTE_ATOMIC);
        return ckernel::semaphore_read(static_cast<std::uint8_t>(handle));
    }
}

// Settled value: retire this thread's own posted SEMPOST/SEMGET or ATINCGET (and the engine work their
// STALLWAITs wait on) before reading.
template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t current(std::uintptr_t index) {
    ckernel::tensix_sync();
    return load<scope>(index);
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void up(std::uintptr_t index, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_COMPUTE_ATOMICS) {
        l1_word_atomic_add(index, value);
    } else {
        static_assert(scope == SemScope::COMPUTE_ATOMIC);
        const std::uint8_t idx = static_cast<std::uint8_t>(index);
        for (std::uint32_t i = 0; i < value; ++i) {
            ckernel::t6_semaphore_post<kEngineIdle>(idx);  // STALLWAIT(my engine) + SEMPOST
        }
    }
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void down(std::uintptr_t index, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_COMPUTE_ATOMICS) {
        l1_word_atomic_add(index, 0u - value);  // no sufficiency wait; pair with wait_min() as for COMPUTE_ATOMIC
    } else {
        static_assert(scope == SemScope::COMPUTE_ATOMIC);
        const std::uint8_t idx = static_cast<std::uint8_t>(index);
        for (std::uint32_t i = 0; i < value; ++i) {
            ckernel::t6_semaphore_get<kEngineIdle>(idx);  // STALLWAIT(my engine) + SEMGET
        }
    }
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC || scope == SemScope::DM_COMPUTE_ATOMICS);
    // No SEMWAIT condition for "== v" (and none at all on an L1 word): RISC poll, after retiring this
    // thread's own posted ops.
    WAYPOINT("NSW");
    ckernel::tensix_sync();
    while (load<scope>(index) != value) {
    }
    WAYPOINT("NSD");
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait_min(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC || scope == SemScope::DM_COMPUTE_ATOMICS);
    if (scope == SemScope::COMPUTE_ATOMIC && value == 1) {
        // Tensix-side: hold this thread's engine instructions while Value == 0. In order with this
        // thread's own SEMGETs, so a just-consumed credit is never counted again. The RISC returns.
        ckernel::t6_semaphore_wait_on_zero<kEngineBlock>(static_cast<std::uint8_t>(index));
        return;
    }
    WAYPOINT("NSMW");
    ckernel::tensix_sync();
    while (load<scope>(index) < value) {
    }
    WAYPOINT("NSMD");
}

// Room for the up(n) that follows. n == 1, Tensix-side: hold this thread's engine instructions while
// Value >= Max (the ring is full). In order with this thread's own SEMPOSTs, so a credit this thread has
// posted but not yet retired still counts. n > 1: SEMWAIT has no "Value <= Max - n" condition, so RISC-poll
// the settled value (as wait()), after retiring this thread's own posted SEMPOSTs.
template <SemScope scope>
__attribute__((always_inline)) inline void wait_not_full(std::uintptr_t index, std::uint32_t n) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC, "wait_not_full() needs the hardware Max: COMPUTE_ATOMIC only");
    if (n == 1) {
        ckernel::t6_semaphore_wait_on_max<kEngineBlock>(static_cast<std::uint8_t>(index));
        return;
    }
    ASSERT(n <= kComputeSemaphoreMax);
    WAYPOINT("NFW");
    ckernel::tensix_sync();
    while (load<scope>(index) > kComputeSemaphoreMax - n) {
    }
    WAYPOINT("NFD");
}

template <SemScope scope>
__attribute__((always_inline)) inline void set(std::uintptr_t index, std::uint32_t value) {
    if constexpr (scope == SemScope::DM_COMPUTE_ATOMICS) {
        // Plain 32-bit ThCon store, ordered after this thread's engine work; destructive, like DM set().
        ckernel::t6_store_l1_32<SemGpr::SEM_ADDR, SemGpr::SEM_DATA, kEngineIdle>(index, value);
    } else {
        static_assert(scope == SemScope::COMPUTE_ATOMIC);
        // SEMINIT takes a 4-bit Value; anything above the capacity would be truncated silently.
        ASSERT(value <= kComputeSemaphoreMax);
        // Absolute assignment via SEMINIT (Max re-programmed to the host-baked capacity), ordered after this
        // thread's engine work for the same reason as up()/down().
        ckernel::t6_semaphore_init<kEngineIdle>(
            static_cast<std::uint8_t>(index), static_cast<std::uint8_t>(value), kComputeSemaphoreMax);
    }
}

#else  // unsupported compute target: non-Blackhole, or the MATH / isolate-SFPU thread

// Declared so the Semaphore class (semaphore.h) type-checks identically on every compute build; each
// rejects on instantiation. A kernel never reaches them: constructing a compute Semaphore already fails
// in sem_l1_offset() off Blackhole, and no compute kernel calls a semaphore method on MATH.
#define COMPUTE_SEMAPHORE_UNSUPPORTED(scope) \
    static_assert(always_false<scope>, "compute semaphores are a Blackhole UNPACK/PACK primitive")
template <ProgrammableCoreType, SemScope scope>
inline void up(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <ProgrammableCoreType, SemScope scope>
inline void down(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <SemScope scope>
inline void wait(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <SemScope scope>
inline void wait_min(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <SemScope scope>
inline void wait_not_full(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <SemScope scope>
inline void set(std::uintptr_t, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
}
template <SemScope scope>
inline std::uint32_t current(std::uintptr_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(scope);
    return 0;
}
#undef COMPUTE_SEMAPHORE_UNSUPPORTED

#endif  // ARCH_BLACKHOLE && (UNPACK || PACK)

}  // namespace semaphore_detail

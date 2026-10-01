// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/semaphore_binding_token.h"  // SemScope + semaphore_detail::always_false
#include "core_config.h"
#include "ckernel.h"
#ifdef ARCH_QUASAR
#include "llk_sync.h"
#endif

// Blackhole and Quasar compute semaphore (SemScope::COMPUTE_ATOMIC): UNPACK <-> PACK synchronization on a
// Tensix hardware (Sync Unit) semaphore, UNPACK_OPERAND_SYNC (index 3) on Blackhole.
//
// Primitives (Value and Max are 4-bit fields of the hardware semaphore):
//   up(n)           n x [ STALLWAIT(my engine idle) ; SEMPOST ]     RISC returns; credit lands after the
//                                                                    packer/unpacker work (publish-after-data)
//   down(n)         n x [ STALLWAIT(my engine idle) ; SEMGET  ]     RISC returns; release after the engine read
//   wait_min(1)     SEMWAIT(block my engine while Value == 0)       RISC returns; holds UNPACR/PACR
//   wait_not_full() SEMWAIT(block my engine while Value >= Max)     RISC returns; producer back-pressure
//   wait_not_full(n>1) settle ; RISC polls Value <= Max - n         RISC blocks; room for a batched up(n)
//   wait(v)         settle ; RISC polls Value == v                   RISC blocks
//   wait_min(v!=1)  settle ; RISC polls Value >= v                   RISC blocks
//   value()         settle ; read Value                              RISC blocks; settled value
//   set(v)          STALLWAIT(my engine idle) ; SEMINIT(Max, v)      RISC returns
// settle (tensix_sync on Blackhole, CSR poll of the Sync busy bit on Quasar) retires this thread's own
// queued SEMPOST/SEMGET first; without it the read can predate them. SEMWAIT has only the "== 0" and
// ">= Max" conditions.
//
// Input / output: Max = COMPUTE_SEMAPHORE_MAX (host-baked from SemaphoreAdvancedOptions::max_value,
// default 15) programmed by every SEMINIT; Value starts at 0, seeded by compute_kernel_hw_startup() on
// PACK, and must be left at 0 by the kernel. Canonical ring use, capacity = ring depth:
//   PACK:   wait_not_full(); pack_tile(ring, slot); up(1);
//   UNPACK: wait_min(1); copy_tile(ring, slot); down(1);
// The wait BEFORE the slot access protects the data; the counter update AFTER it only keeps the count.
// Skipping wait_not_full() lets PACK overwrite a slot UNPACK has not read yet; skipping wait_min(1) lets
// UNPACK read a slot PACK has not written yet.
//
// Limits: Value 0..15; Blackhole SEMPOST saturates at 15 (not Max), SEMGET floors silently;
// Quasar SEMPOST stalls at Max and SEMGET stalls at 0 (see below). Core-local (no NoC, no DM core);
// one hardware semaphore is reserved for it, so one compute semaphore per program (host-enforced);
// waits order this thread's engine instructions, not RISC L1 loads (use value() to gate a RISC read).
// Design rationale and measurements: PR #56189.
//
// Quasar differences (primitives in semaphore_detail::hw below):
//  - The Sync Unit applies back-pressure: up() stalls at Max until a SEMGET makes room, and down() stalls
//    at 0 until a SEMPOST arrives, holding back the thread's later instructions (PACR/UNPACR included).
//    Nothing is dropped, unlike Blackhole, so a count mismatch hangs instead of losing credits. This
//    keeps the count right but cannot replace either wait: it happens after the slot access.
//  - settle() is a CSR poll of the Sync busy bit (wait_sync_idle()) instead of tensix_sync.

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

// The bound id selects nothing here: every compute semaphore is the one free Sync Unit index. The
// function keeps the name sem_l1_offset so the shared Semaphore class template (semaphore.h) needs no
// change; the returned value is the hardware index, threaded through up/down/wait/... below.
template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline std::uintptr_t sem_l1_offset(std::uint32_t /*id*/, SemScope scope) {
    static_assert(core_type == ProgrammableCoreType::TENSIX, "compute semaphores require a Tensix core");
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);  // compute supports COMPUTE_ATOMIC only
#if defined(ARCH_BLACKHOLE)
    return ckernel::semaphore::UNPACK_OPERAND_SYNC;
#elif defined(ARCH_QUASAR)
    return ckernel::trisc::semaphore::PACK_UNPACK;
#else
    // Other archs' LLKs do not define this index; fail only if a kernel actually constructs one.
    static_assert(always_false<core_type>, "compute semaphores are Blackhole and Quasar only");
    return 0;
#endif
}

#if (defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)) && (defined(TRISC_UNPACK) || defined(TRISC_PACK))

// Per-arch Tensix primitives. post/get are STALLWAIT(my engine idle) + SEMPOST/SEMGET, so up()/down() are
// ordered after this thread's engine work. wait_on_zero/wait_on_max are a SEMWAIT that blocks only this
// thread's engine instructions. settle() retires this thread's queued Sync Unit ops before a RISC read.
namespace hw {

#if defined(ARCH_QUASAR)

// STALLWAIT takes wait-resource indices (up to three), so name every unit of this thread's engine.
#if defined(TRISC_UNPACK)
inline constexpr std::uint32_t kIdle0 = ckernel::p_stall::UNPACK0;
inline constexpr std::uint32_t kIdle1 = ckernel::p_stall::UNPACK1;
inline constexpr std::uint32_t kIdle2 = ckernel::p_stall::UNPACK2;
inline constexpr std::uint32_t kBlock = ckernel::p_stall::STALL_UNPACK;
#else
inline constexpr std::uint32_t kIdle0 = ckernel::p_stall::PACK0;
inline constexpr std::uint32_t kIdle1 = ckernel::p_stall::PACK1;
inline constexpr std::uint32_t kIdle2 = ckernel::p_stall::NOTHING;
inline constexpr std::uint32_t kBlock = ckernel::p_stall::STALL_PACK;
#endif

__attribute__((always_inline)) inline void post(std::uint8_t idx) { _llk_sync_post_<kIdle0, kIdle1, kIdle2>(idx); }
__attribute__((always_inline)) inline void get(std::uint8_t idx) { _llk_sync_get_<kIdle0, kIdle1, kIdle2>(idx); }
__attribute__((always_inline)) inline void wait_on_zero(std::uint8_t idx) {
    _llk_sync_wait_<kBlock, ckernel::p_stall::STALL_ON_ZERO>(idx);
}
__attribute__((always_inline)) inline void wait_on_max(std::uint8_t idx) {
    _llk_sync_wait_<kBlock, ckernel::p_stall::STALL_ON_MAX>(idx);
}
__attribute__((always_inline)) inline void init(std::uint8_t idx, std::uint8_t value) {
    ckernel::trisc::t6_semaphore_init<kIdle0, kIdle1, kIdle2>(idx, value, kComputeSemaphoreMax);
}
// CSR poll of the Sync busy bit: clear once this thread's SEMPOST/SEMGET (queued or in flight) retired.
__attribute__((always_inline)) inline void settle() { ckernel::wait_sync_idle(); }

#else  // ARCH_BLACKHOLE

#if defined(TRISC_UNPACK)
inline constexpr std::uint32_t kIdle = ckernel::p_stall::UNPACK;         // C1|C2: both unpackers idle
inline constexpr std::uint32_t kBlock = ckernel::p_stall::STALL_UNPACK;  // B3: block UNPACR
#else
inline constexpr std::uint32_t kIdle = ckernel::p_stall::PACK;         // C3: packer idle
inline constexpr std::uint32_t kBlock = ckernel::p_stall::STALL_PACK;  // B2: block PACR
#endif

__attribute__((always_inline)) inline void post(std::uint8_t idx) { ckernel::t6_semaphore_post<kIdle>(idx); }
__attribute__((always_inline)) inline void get(std::uint8_t idx) { ckernel::t6_semaphore_get<kIdle>(idx); }
__attribute__((always_inline)) inline void wait_on_zero(std::uint8_t idx) {
    ckernel::t6_semaphore_wait_on_zero<kBlock>(idx);
}
__attribute__((always_inline)) inline void wait_on_max(std::uint8_t idx) {
    ckernel::t6_semaphore_wait_on_max<kBlock>(idx);
}
__attribute__((always_inline)) inline void init(std::uint8_t idx, std::uint8_t value) {
    ckernel::t6_semaphore_init<kIdle>(idx, value, kComputeSemaphoreMax);
}
__attribute__((always_inline)) inline void settle() { ckernel::tensix_sync(); }

#endif  // ARCH_QUASAR

}  // namespace hw

__attribute__((always_inline)) inline std::uint32_t load(std::uintptr_t index, SemScope scope) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    return ckernel::semaphore_read(static_cast<std::uint8_t>(index));
}

// Settled value: retire this thread's own posted SEMPOST/SEMGET (and the engine work their STALLWAITs
// wait on) before reading.
template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline std::uint32_t current(std::uintptr_t index, SemScope scope) {
    hw::settle();
    return load(index, scope);
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void up(std::uintptr_t index, SemScope scope, std::uint32_t value) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    const std::uint8_t idx = static_cast<std::uint8_t>(index);
    for (std::uint32_t i = 0; i < value; ++i) {
        hw::post(idx);
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void down(std::uintptr_t index, SemScope scope, std::uint32_t value) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    const std::uint8_t idx = static_cast<std::uint8_t>(index);
    for (std::uint32_t i = 0; i < value; ++i) {
        hw::get(idx);
    }
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void wait(std::uintptr_t index, SemScope scope, std::uint32_t value) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    // No SEMWAIT condition for "== v": RISC poll, after retiring this thread's own posted ops.
    WAYPOINT("NSW");
    hw::settle();
    while (load(index, scope) != value) {
    }
    WAYPOINT("NSD");
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void wait_min(std::uintptr_t index, SemScope scope, std::uint32_t value) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    if (value == 1) {
        // Tensix-side: hold this thread's engine instructions while Value == 0. In order with this
        // thread's own SEMGETs, so a just-consumed credit is never counted again. The RISC returns.
        hw::wait_on_zero(static_cast<std::uint8_t>(index));
        return;
    }
    WAYPOINT("NSMW");
    hw::settle();
    while (load(index, scope) < value) {
    }
    WAYPOINT("NSMD");
}

// Room for the up(n) that follows. n == 1, Tensix-side: hold this thread's engine instructions while
// Value >= Max (the ring is full). In order with this thread's own SEMPOSTs, so a credit this thread has
// posted but not yet retired still counts. n > 1: SEMWAIT has no "Value <= Max - n" condition, so RISC-poll
// the settled value (as wait()), after retiring this thread's own posted SEMPOSTs.
template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void wait_not_full(std::uintptr_t index, SemScope scope, std::uint32_t n) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    if (n == 1) {
        hw::wait_on_max(static_cast<std::uint8_t>(index));
        return;
    }
    ASSERT(n <= kComputeSemaphoreMax);
    WAYPOINT("NFW");
    hw::settle();
    while (load(index, scope) > kComputeSemaphoreMax - n) {
    }
    WAYPOINT("NFD");
}

template <ProgrammableCoreType core_type>
__attribute__((always_inline)) inline void set(std::uintptr_t index, SemScope scope, std::uint32_t value) {
    ASSERT(scope == SemScope::COMPUTE_ATOMIC);
    // SEMINIT takes a 4-bit Value; anything above the capacity would be truncated silently.
    ASSERT(value <= kComputeSemaphoreMax);
    // Absolute assignment via SEMINIT (Max re-programmed to the host-baked capacity), ordered after this
    // thread's engine work for the same reason as up()/down().
    hw::init(static_cast<std::uint8_t>(index), static_cast<std::uint8_t>(value));
}

#else  // unsupported compute target: not Blackhole/Quasar, or the MATH / isolate-SFPU thread

// Declared so the Semaphore class (semaphore.h) type-checks identically on every compute build; each
// rejects on instantiation. A kernel never reaches them: constructing a compute Semaphore already fails
// in sem_l1_offset() off Blackhole/Quasar, and no compute kernel calls a semaphore method on MATH.
#define COMPUTE_SEMAPHORE_UNSUPPORTED(core_type) \
    static_assert(always_false<core_type>, "compute semaphores are a Blackhole/Quasar UNPACK/PACK primitive")
template <ProgrammableCoreType core_type>
inline void up(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline void down(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline void wait(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline void wait_min(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline void wait_not_full(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline void set(std::uintptr_t, SemScope, std::uint32_t) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
}
template <ProgrammableCoreType core_type>
inline std::uint32_t current(std::uintptr_t, SemScope) {
    COMPUTE_SEMAPHORE_UNSUPPORTED(core_type);
    return 0;
}
#undef COMPUTE_SEMAPHORE_UNSUPPORTED

#endif  // (ARCH_BLACKHOLE || ARCH_QUASAR) && (UNPACK || PACK)

}  // namespace semaphore_detail

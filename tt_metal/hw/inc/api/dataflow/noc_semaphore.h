// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"

/**
 * @brief Physical path a semaphore's accesses take. This is picked by the host at compile time
 *        and baked into the kernel. The path picked provides the fastest access that keeps the
 *        semaphore operations atomic.
 *
 *  - LOCAL_NONATOMIC: L1 read-modify-write. Picked only when at most one binder instance exists.
 *  - DM_LOCAL_CACHED: Touched only by DM threads on the semaphore's one node -- any number of
 *      binder kernels/threads (the node's DM harts are mutually coherent). Increments are a
 *      32-bit RISC-V AMO and down() an LR/SC conditional decrement, both on the CACHED alias.
 *      The word lives in a dedicated cached-only pool that the NoC path never addresses; the
 *      host census guarantees no NoC or remote access exists (runtime watcher ASSERTs back it).
 *      Generated entry/exit stubs seed the pool row once per program and self-restore it when
 *      the last binder hart exits (see genfiles.cpp).
 *  - EXTERNAL: reachable over the NoC. All ops go through self-targeted NoC atomics; down() is
 *      multi-consumer-safe on Quasar DM via a NoC-CAS lock. Values must never legitimately be
 *      0xFFFFFFFF (the CAS-return sentinel).
 *
 * The host-side mirror of this enum lives in jit_build/jit_build_settings.hpp; the generated
 * per-kernel scope table (numeric values) is the bridge. Two emitted tripwires guard it
 * (kernel_bindings_generated.h): one static_asserts this enum's numbering against the host
 * mirror's values, one static_asserts per bound id that the table was visible when this header
 * was compiled (include-order regression).
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
};

// The host's chosen mechanism for each bound semaphore id, injected invisibly by codegen:
// kernel_bindings_generated.h #defines TT_METAL2_SEM_SCOPE_TABLE before this header is ever
// seen in a Metal 2.0 kernel TU. Unbound ids -- and every id in a legacy kernel, which has no
// generated header -- resolve to LOCAL_NONATOMIC, the historical plain-word behavior. For a
// compile-time id (`sem::x` is constexpr) the lookup and the mechanism dispatch below fold
// away entirely; a genuinely runtime id keeps a small predictable branch and stays correct.
inline __attribute__((always_inline)) constexpr SemScope sem_scope_of(uint32_t semaphore_id) {
#ifdef TT_METAL2_SEM_SCOPE_TABLE
    constexpr SemScope table[] = TT_METAL2_SEM_SCOPE_TABLE;
    constexpr uint32_t n = sizeof(table) / sizeof(table[0]);
    return semaphore_id < n ? table[semaphore_id] : SemScope::LOCAL_NONATOMIC;
#else
    (void)semaphore_id;
    return SemScope::LOCAL_NONATOMIC;
#endif
}

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The Semaphore class provides a simple interface for semaphore-based synchronization
 * between programmable cores. It allows incrementing and decrementing the semaphore value,
 * as well as waiting for the semaphore to reach a desired value. The semaphore can be
 * manipulated locally or remotely via the NoC.
 *
 * The physical mechanism is host-chosen and invisible (see SemScope / sem_scope_of above):
 * construct with the semaphore id and call the methods.
 *
 * Usage:
 *   - Construct a Semaphore with a given semaphore ID.
 *   - Use up(), down(), and other methods to perform synchronization.
 *
 * Methods:
 *  - up(value): Increment the semaphore by the specified value locally.
 *  - up(value, noc_x, noc_y, noc, vc): Atomically increment the semaphore by the specified value on a remote core.
 *  - down(value): Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
 *
 * The following methods (non-standard semantics) are also available, for parity with existing API:
 *  - wait(value): Block until the semaphore is set to the specified value.  Does not decrement the semaphore.
 *  - wait_min(value): Block until the semaphore is at least the specified value.  Does not decrement the semaphore.
 *  - set(value): Set the semaphore to the specified value.
 *  - value(): Read the current semaphore value (via this scope's coherent view).
 *  - set_multicast(...): Set the semaphore value on multiple cores.
 *  - set_multicast_loopback_src(...): Set the semaphore value on multiple cores including the source.
 *  - relay_unicast(dst_sem, ...): Set a different remote semaphore on one core to this semaphore's local value.
 *  - relay_multicast(dst_sem, ...): Multicast this semaphore's local value into a different destination semaphore.
 */
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
class Semaphore {
    // Lets relay_unicast / relay_multicast read dst_sem's private members without a public accessor.
    template <ProgrammableCoreType OT>
    friend class Semaphore;

    // LOCAL_NONATOMIC and EXTERNAL access the local word through the uncached alias on Quasar
    // (coherent with NoC atomics landing at TL1). DM_LOCAL_CACHED uses the cached alias (AMOs
    // hang on the uncached alias; DM cores are mutually coherent on the cached one).
    // Each scope keeps its word in exactly one view, so no flush or invalidate is required
    // here; the one mandatory alias-discipline step lives in the generated pool seeder
    // (genfiles.cpp), which reads the ring slot through the UNCACHED alias.
    // LOAD-BEARING: the invalidate_l1_cache() calls below (and in noc_semaphore_wait/wait_min)
    // are a documented NO-OP on Quasar DM cores (tt-2xx/risc_common.h). If that ever becomes a
    // real discard-without-writeback, a cached semaphore would silently lose increments.

    // Physical L1 offset of semaphore `id`: DM_LOCAL_CACHED lives in the dedicated cached-only
    // pool (MEM_DM_CACHED_SEM_BASE); every other scope uses the normal ring (get_semaphore).
    // MEM_L1_BASE == 0, so the offset is also the cached-alias address.
    static __attribute__((always_inline)) inline uintptr_t sem_l1_offset(uint32_t id) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        if (sem_scope_of(id) == SemScope::DM_LOCAL_CACHED) {
            ASSERT(id < MEM_DM_CACHED_SEM_SIZE / MEM_DM_CACHED_SEM_ROW);
            return static_cast<uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * MEM_DM_CACHED_SEM_ROW;
        }
#endif
        return get_semaphore<core_type>(id);
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC) && !defined(TT_EMULE_USE_L1_POOL)
    // This EXTERNAL semaphore's lock word on THIS node (firmware-zeroed at boot; only ever 0/1).
    // One lock per 16B row, so the 4-bit NoC-CAS always addresses lane 0 of its row (see
    // MEM_NOC_SEM_LOCK_SIZE in dev_mem_map.h for why lane 0).
    uint32_t external_lock_l1_offset() const {
        const uint32_t id =
            (static_cast<uint32_t>(l1_offset_) - static_cast<uint32_t>(get_semaphore<core_type>(0))) / L1_ALIGNMENT;
        ASSERT(id * L1_ALIGNMENT < MEM_NOC_SEM_LOCK_SIZE);
        return MEM_NOC_SEM_LOCK_BASE + id * L1_ALIGNMENT;
    }
    // This hart's private CAS-return slot (R_SRC_ADDR is per-hart sticky).
    static uint32_t cas_ret_slot() {
        uint64_t hart;
        asm volatile("csrr %0, mhartid" : "=r"(hart));
        ASSERT(static_cast<uint32_t>(hart) * 4 < MEM_NOC_CAS_RET_SIZE);
        return MEM_NOC_CAS_RET_BASE + static_cast<uint32_t>(hart) * 4;
    }
#endif

public:
    // l1_offset_ holds the physical L1 offset of the semaphore word (the cached-alias
    // address; MEM_L1_BASE == 0). Local views and NoC addresses are derived from it.
    explicit __attribute__((always_inline)) Semaphore(uint32_t semaphore_id) :
        l1_offset_(sem_l1_offset(semaphore_id)), scope_(sem_scope_of(semaphore_id)) {}

    /**
     * @brief Increment the semaphore by the specified value (local).
     *
     * DM_LOCAL_CACHED: atomic 32-bit AMO on the cached alias.
     * EXTERNAL:        self-targeted NoC atomic increment (local + remote writers serialize at one NIU).
     * LOCAL_NONATOMIC: plain L1 read-modify-write (NOT atomic; legacy default).
     *
     * @param value The value to increment the semaphore by.
     */
    __attribute__((always_inline)) void up(uint32_t value) {
        if (scope_ == SemScope::DM_LOCAL_CACHED) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
            __atomic_add_fetch(reinterpret_cast<uint32_t*>(l1_offset_), value, __ATOMIC_SEQ_CST);
#else
            ASSERT(false);  // the host census never bakes CACHED for this platform
#endif
        } else if (scope_ == SemScope::EXTERNAL) {
#ifndef COMPILE_FOR_TRISC
            noc_semaphore_inc(::get_noc_addr(l1_offset_), value);
            noc_async_atomic_barrier();
#else
            ASSERT(false);  // compute kernels cannot bind semaphores (host-rejected)
#endif
        } else {  // LOCAL_NONATOMIC
            *local_ptr() += value;
        }
    }

    /**
     * @brief Atomically increment the semaphore by the specified value on a remote core.
     *
     * Not valid on a DM_LOCAL_CACHED semaphore (its word must never be touched via the NoC);
     * the host census never picks CACHED for a semaphore with a remote writer.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @param value The value to increment the semaphore by.
     * @param vc The virtual channel to use for the transaction (default is NOC_UNICAST_WRITE_VC).
     */
    __attribute__((always_inline)) void up(
        const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t value, uint8_t vc = NOC_UNICAST_WRITE_VC) {
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);
        const uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, noc.get_noc_id());
        noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
    }

    /**
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     *
     * DM_LOCAL_CACHED: LR/SC conditional-decrement retry loop — MULTI-CONSUMER-SAFE.
     * EXTERNAL (Quasar DM): MULTI-CONSUMER-SAFE — a 4-bit NoC-CAS spinlock on the semaphore's
     *   lock word guards the >=value check + INCR_GET subtract; consumers must run on the
     *   semaphore's node. Gen1 and emule (TT_EMULE_USE_L1_POOL) keep the historical
     *   single-consumer spin+subtract — the caller owns that invariant there.
     * INVARIANT (EXTERNAL, Quasar): the semaphore value must never legitimately be 0xFFFFFFFF —
     *   it is the CAS-return sentinel.
     * LOCAL_NONATOMIC: legacy single-owner (non-atomic) decrement after an uncached spin.
     *
     * @param value The value to decrement the semaphore by.
     */
    __attribute__((always_inline)) void down(uint32_t value) {
        auto* sem_addr = local_ptr();
        WAYPOINT("NSDW");
        if (scope_ == SemScope::DM_LOCAL_CACHED) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
            // Conditional decrement via LR/SC strong CAS on the CACHED alias (LR/SC and AMOs hang
            // uncached). A losing CAS re-enters the >= check, so two consumers can never both take
            // the same credit. A plain AMO subtract after the spin would NOT be multi-consumer-safe.
            auto* word = reinterpret_cast<uint32_t*>(l1_offset_);  // cached alias
            uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
            do {
                // Re-arm each retry so a starved consumer dumps as waiting (NSDW), not NSDD.
                WAYPOINT("NSDW");
                while (observed < value) {  // DM cores are mutually coherent; no invalidate needed
                    observed = __atomic_load_n(word, __ATOMIC_RELAXED);
                }
                WAYPOINT("NSDD");
            } while (!__atomic_compare_exchange_n(
                word, &observed, observed - value, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
#else
            ASSERT(false);  // the host census never bakes CACHED for this platform
#endif
        } else if (scope_ == SemScope::EXTERNAL) {
            // TT_EMULE_USE_L1_POOL: emule compiles the Gen1 arm (its shims cover those primitives).
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC) && !defined(TT_EMULE_USE_L1_POOL)
            // Lock protocol and sentinel invariant: see the down() doc block. Producers bypass the
            // lock safely: CAS and INCR_GET serialize mutually at the NIU (TestSelfCasLockVsIncr).
            // Every atomic's pre-op return lands at this hart's sticky ret slot and is consumed by
            // a sentinel pre-write + poll.
            // Drain any prior in-flight atomic FIRST (e.g. a remote up() does not barrier): its
            // pre-op return landing after our acquire CAS's would corrupt the lock verdict.
            noc_async_atomic_barrier();
            const uint64_t sem_noc = ::get_noc_addr(l1_offset_);
            const uint64_t lock_noc = ::get_noc_addr(external_lock_l1_offset());
            const uint32_t ret_slot = cas_ret_slot();
            auto* ret_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) + ret_slot);
            constexpr uint32_t kCasSentinel = 0xFFFFFFFFu;
            // One consumed NoC atomic: sentinel, issue, fence, poll the returned pre-op value.
            auto consumed_cas4 = [&](uint32_t cmp4, uint32_t swap4) -> uint32_t {
                *ret_word = kCasSentinel;
                noc_fast_atomic_cas4<DM_DEDICATED_NOC>(
                    noc_index, lock_noc, NOC_UNICAST_WRITE_VC, cmp4, swap4, ret_slot);
                noc_async_atomic_barrier();
                // Re-arm: the barrier's waypoints overwrote ours; a hang here must dump as NSDD.
                WAYPOINT("NSDD");
                while (*ret_word == kCasSentinel) {
                }
                return *ret_word;
            };
            for (;;) {
                // Lock-free pre-wait: don't contend the lock until credit is plausibly there.
                WAYPOINT("NSDW");
                do {
                    invalidate_l1_cache();
                } while ((*sem_addr) < value);
                if (consumed_cas4(/*cmp4=*/0, /*swap4=*/1) != 0) {
                    continue;  // contended: back to the lock-free wait
                }
                // Re-check under the lock: no other consumer can decrement now; producers only add.
                invalidate_l1_cache();
                const bool ok = (*sem_addr) >= value;
                if (ok) {
                    WAYPOINT("NSDD");
                    // The atomic subtract (INCR_GET of the two's complement, wrap=31). Its pre-op
                    // return also lands at ret_slot; consume it before the release CAS's sentinel.
                    *ret_word = kCasSentinel;
                    noc_semaphore_inc(sem_noc, (uint32_t)(0u - value));
                    noc_async_atomic_barrier();
                    WAYPOINT("NSDD");  // re-arm past the barrier's internal waypoints
                    // BOUNDED poll: the barrier already orders the return write (keystone-pinned:
                    // TestAtomicCasReturnsPreOpValue), so this exits on its first check. The bound
                    // exists so a semaphore that VIOLATES the 0xFFFFFFFF invariant (its pre-op value
                    // equals the sentinel) cannot spin forever inside the lock and wedge every other
                    // consumer of this id; the subtract itself completed at the barrier either way.
                    for (uint32_t spin = 0; spin < 1024u && *ret_word == kCasSentinel; spin++) {
                    }
                }
                consumed_cas4(/*cmp4=*/1, /*swap4=*/0);  // release (holder-only, always succeeds)
                if (ok) {
                    return;
                }
                // Credit vanished before we locked: released; back to the lock-free wait.
            }
#elif !defined(COMPILE_FOR_TRISC)
            // Gen1 single-consumer path: spin, then atomic subtract.
            do {
                invalidate_l1_cache();
            } while ((*sem_addr) < value);
            WAYPOINT("NSDD");
            noc_semaphore_inc(::get_noc_addr(l1_offset_), (uint32_t)(0u - value));
            noc_async_atomic_barrier();
#else
            ASSERT(false);  // compute kernels cannot bind semaphores (host-rejected)
#endif
        } else {  // LOCAL_NONATOMIC (legacy)
            do {
                invalidate_l1_cache();
            } while ((*sem_addr) < value);
            WAYPOINT("NSDD");
            *sem_addr -= value;
        }
    }

    // The following methods provide parity with existing semaphore API, but have non-standard semantics.

    /**
     * @brief Block until the semaphore is set to the specified value.
     *
     * @param value The value to wait for.
     */
    __attribute__((always_inline)) void wait(uint32_t value) { noc_semaphore_wait(local_ptr(), value); }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    __attribute__((always_inline)) void wait_min(uint32_t value) { noc_semaphore_wait_min(local_ptr(), value); }

    /**
     * @brief Set the semaphore to the specified value.
     *
     * A non-atomic destructive store under every scope: use it for init/reset only, never
     * concurrently with up()/down().
     *
     * @param value The value to set the semaphore to.
     */
    __attribute__((always_inline)) void set(uint32_t value) { noc_semaphore_set(local_ptr(), value); }

    /**
     * @brief Read the current semaphore value through this scope's coherent view.
     */
    __attribute__((always_inline)) uint32_t value() const {
        invalidate_l1_cache();
        return *local_ptr();
    }

    /**
     * @brief Relay this semaphore's local value into a different remote semaphore on a single core.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). To bump the
     *       same semaphore on a remote core, use up(noc, noc_x, noc_y, value) instead.
     *       Writes 4 bytes from this->l1_offset_ to dst_sem.l1_offset_ on the remote core
     *       (noc_x, noc_y).
     *       Neither endpoint may be DM_LOCAL_CACHED (a relay is a NoC write; the cached pool
     *       must never be NoC-written) -- guarded by the runtime ASSERT below (the census sees topology, not method
     * calls).
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param dst_sem The destination Semaphore whose L1 offset receives the value.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @tparam dst_core_type Programmable core type of the destination (defaults to this Semaphore's core_type).
     */
    template <ProgrammableCoreType dst_core_type = core_type>
    void relay_unicast(const Noc& noc, const Semaphore<dst_core_type>& dst_sem, uint32_t noc_x, uint32_t noc_y) {
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);
        ASSERT(dst_sem.scope_ != SemScope::DM_LOCAL_CACHED);
        ASSERT(l1_offset_ != dst_sem.l1_offset_);
        const uint64_t dst_noc_addr = ::get_noc_addr(noc_x, noc_y, dst_sem.get_l1_addr(), noc.get_noc_id());
        noc_semaphore_set_remote(get_l1_addr(), dst_noc_addr, noc.get_noc_id());
    }

    /**
     * @brief Set the semaphore value on multiple cores in a specified rectangular region of the NoC.
     * @note Sender cannot be part of the multicast destinations.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param noc_x_start The starting X coordinate of the region (inclusive).
     * @param noc_y_start The starting Y coordinate of the region (inclusive).
     * @param noc_x_end The ending X coordinate of the region (inclusive).
     * @param noc_y_end The ending Y coordinate of the region (inclusive).
     * @param num_dests The number of destination cores in the region.
     * @param linked Whether to link this operation with the next (default is false).
     * @tparam opts NocOptions flags; set NocOptions::MCAST_INCL_SRC to include the sender in the multicast
     *             (default is NocOptions::DEFAULT which excludes sender)
     */
    template <NocOptions opts = NocOptions::DEFAULT>
    void set_multicast(
        const Noc& noc,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);  // a multicast set is a NoC write
        const uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, noc.get_noc_id());
        const uintptr_t src_l1_addr = get_l1_addr();
        if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
            noc_semaphore_set_multicast_loopback_src(src_l1_addr, multicast_addr, num_dests, linked, noc.get_noc_id());
        } else {
            noc_semaphore_set_multicast(src_l1_addr, multicast_addr, num_dests, linked, noc.get_noc_id());
        }
    }

    /**
     * @brief Relay this semaphore's local value into a different destination semaphore on a rectangular region.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). Each core in the region
     *       receives the 4-byte write at dst_sem's L1 offset.
     * @note Sender cannot be part of the multicast destinations unless mcast_mode is INCLUDE_SRC.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param dst_sem The destination Semaphore whose L1 offset receives the value on each core in the region.
     * @param noc_x_start The starting X coordinate of the region (inclusive).
     * @param noc_y_start The starting Y coordinate of the region (inclusive).
     * @param noc_x_end The ending X coordinate of the region (inclusive).
     * @param noc_y_end The ending Y coordinate of the region (inclusive).
     * @param num_dests The number of destination cores in the region.
     * @param linked Whether to link this operation with the next (default is false).
     * @tparam opts NocOptions flags; set NocOptions::MCAST_INCL_SRC to include the sender in the multicast
     *             (default is NocOptions::DEFAULT which excludes sender)
     * @tparam dst_core_type Programmable core type of the destination (defaults to this Semaphore's core_type).
     */
    template <NocOptions opts = NocOptions::DEFAULT, ProgrammableCoreType dst_core_type = core_type>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type>& dst_sem,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);
        ASSERT(dst_sem.scope_ != SemScope::DM_LOCAL_CACHED);
        ASSERT(l1_offset_ != dst_sem.l1_offset_);
        const uint64_t multicast_addr = ::get_noc_multicast_addr(
            noc_x_start, noc_y_start, noc_x_end, noc_y_end, dst_sem.get_l1_addr(), noc.get_noc_id());
        const uintptr_t src_l1_addr = get_l1_addr();
        if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
            noc_semaphore_set_multicast_loopback_src(src_l1_addr, multicast_addr, num_dests, linked, noc.get_noc_id());
        } else {
            noc_semaphore_set_multicast(src_l1_addr, multicast_addr, num_dests, linked, noc.get_noc_id());
        }
    }

    /**
     * @brief Atomically increment the semaphore value on multiple cores in a specified rectangular region of the NoC.
     * @note Sender cannot be part of the multicast destinations.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param noc_x_start The starting X coordinate of the region (inclusive).
     * @param noc_y_start The starting Y coordinate of the region (inclusive).
     * @param noc_x_end The ending X coordinate of the region (inclusive).
     * @param noc_y_end The ending Y coordinate of the region (inclusive).
     * @param value The value to increment the semaphore by.
     * @param num_dests The number of destination cores in the region.
     */
    void inc_multicast(
        const Noc& noc,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t value,
        uint32_t num_dests) {
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);  // a multicast increment is a NoC write
        const uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, noc.get_noc_id());
        noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
    }

private:
    uintptr_t l1_offset_;  // physical L1 offset of the semaphore word (cached-alias address)
    SemScope scope_;       // host-chosen mechanism (from the invisible codegen table)

    // Local access pointer for reads / non-atomic writes. Uncached alias on Quasar for
    // LOCAL_NONATOMIC and EXTERNAL; cached alias for DM_LOCAL_CACHED.
    __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* local_ptr() const {
        uintptr_t addr = l1_offset_;
#ifdef ARCH_QUASAR
        if (scope_ != SemScope::DM_LOCAL_CACHED) {
            addr += MEM_L1_UNCACHED_BASE;
        }
#endif
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    // Physical L1 offset used to form NoC addresses (the NoC addresses TL1 directly).
    uintptr_t get_l1_addr() const { return l1_offset_; }

    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint8_t noc) const {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, get_l1_addr(), noc);
    }

    uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, get_l1_addr(), noc);
    }
};

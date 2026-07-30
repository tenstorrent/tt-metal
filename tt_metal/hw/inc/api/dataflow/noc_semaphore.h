// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include <hostdevcommon/sem_scope.h>       // enum class SemScope (shared host/device, Phase-2 baking)
#include "api/dataflow/semaphore_token.h"  // SemAccessor<Id,Scope> baked-sem token (Phase-2 S1)

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The Semaphore class provides a simple interface for semaphore-based synchronization
 * between programmable cores. It allows incrementing and decrementing the semaphore value,
 * as well as waiting for the semaphore to reach a desired value. The semaphore can be
 * manipulated locally or remotely via the NoC.
 *
 * The Scope template parameter selects the physical path / atomicity guarantee — see
 * SemScope above. It defaults to LOCAL_NONATOMIC (legacy behavior).
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
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX, SemScope Scope = SemScope::LOCAL_NONATOMIC>
class Semaphore {
    // Lets relay_unicast / relay_multicast read dst_sem's private members without a public accessor.
    template <ProgrammableCoreType OT, SemScope OS>
    friend class Semaphore;

    // LOCAL_NONATOMIC and EXTERNAL access the local word through the uncached alias on
    // Quasar (so DM reads/writes are coherent with NoC atomics landing at TL1).
    // DM_LOCAL_CACHED uses the cached alias (RISC-V AMOs require the cache and hang on
    // the uncached alias; DM cores are mutually coherent on the cached alias).
    // ---- Cache discipline (why there are no flushes or invalidates in here) ----
    //
    // Each scope keeps a semaphore word in EXACTLY ONE view, so no party ever needs to reconcile two:
    //   LOCAL_NONATOMIC / EXTERNAL: ring word, always through the UNCACHED alias. Plain stores and NoC
    //     atomics both land at TL1 and uncached reads observe TL1 -- the cache is not in the path.
    //   DM_LOCAL_CACHED: pool word, always through the CACHED alias -- up(), down()'s spin AND its
    //     subtract, wait(), wait_min(), set() and value() all resolve to the same pool address. Every
    //     reader and writer is a DM core, and Quasar DM cores are mutually coherent, so a flush would
    //     publish to nobody and an invalidate would only risk discarding a live count.
    // The pool is cached-only BY CONSTRUCTION (nothing NoC-writes it), which is what makes "no flush"
    // correct rather than merely convenient; it also makes cross-program staleness a non-issue, since
    // the next program's seed is itself a cached store in the same coherent domain and simply wins.
    //
    // The one cache op that IS mandatory lives in the generated pool seeder (genfiles.cpp): it reads the
    // ring slot through the UNCACHED alias, because the dispatcher NoC-wrote that value and a cached
    // read could return stale data.
    //
    // LOAD-BEARING ASSUMPTION: invalidate_l1_cache() is a documented NO-OP on Quasar DM cores
    // (tt-2xx/risc_common.h). The calls below (and inside noc_semaphore_wait/wait_min) are therefore
    // harmless on a cached semaphore. If that function ever becomes a real discard-without-writeback,
    // those call sites would throw away a DIRTY pool line and silently lose increments -- so a cached
    // semaphore must then read through a path that does not invalidate.
    static constexpr bool kUseUncachedLocalView = (Scope != SemScope::DM_LOCAL_CACHED);

    // Physical L1 offset of semaphore `id` for THIS scope. DM_LOCAL_CACHED semaphores live in
    // the dedicated cached-only pool (MEM_DM_CACHED_SEM_BASE, disjoint from the NoC-written
    // kernel_config ring by construction); every other scope uses the normal ring region
    // (sem_l1_base + id*L1_ALIGNMENT via get_semaphore). MEM_L1_BASE == 0, so the returned
    // offset is also the cached-alias address.
    static uintptr_t sem_l1_offset(uint32_t id) {
#ifdef ARCH_QUASAR
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            return static_cast<uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * L1_ALIGNMENT;
        }
#endif
        return get_semaphore<core_type>(id);
    }

public:
    // l1_offset_ holds the physical L1 offset of the semaphore word (the cached-alias
    // address; MEM_L1_BASE == 0). Local views and NoC addresses are derived from it.
    explicit Semaphore(uint32_t semaphore_id) : l1_offset_(sem_l1_offset(semaphore_id)) {}

    // Construct from a host-baked sem:: accessor token (Phase-2). The Scope comes from
    // the token via CTAD (deduction guide after the class), so `Semaphore s(sem::x);`
    // picks the baked mechanism with no explicit <>. The static_assert catches the
    // mismatch case: an explicit `Semaphore<..., WrongScope>(sem::x)` fails to compile
    // instead of silently overriding the baked scope.
    template <uint32_t Id, SemScope TokenScope>
    explicit Semaphore(SemAccessor<Id, TokenScope>) : l1_offset_(sem_l1_offset(Id)) {
#ifdef ARCH_QUASAR
        // The cached-only pool is indexed by the semaphore's own id, so a high id must not run past
        // the pool into whatever follows it. Checkable here (unlike on the host) because both the id
        // and the pool size are visible.
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            static_assert(
                Id * L1_ALIGNMENT < MEM_DM_CACHED_SEM_SIZE,
                "semaphore id does not fit the DM cached semaphore pool (grow MEM_DM_CACHED_SEM_SIZE)");
        }
#endif
        static_assert(
            TokenScope == Scope,
            "The sem:: accessor's baked SemScope does not match this Semaphore's Scope. Write "
            "`Semaphore s(sem::x);` and let CTAD pick the baked scope; do not spell "
            "`Semaphore<...>` explicitly on a baked accessor.");
    }

    /**
     * @brief Increment the semaphore by the specified value (local).
     *
     * DM_LOCAL_CACHED: atomic 32-bit AMO on the cached alias.
     * EXTERNAL:        self-targeted NoC atomic increment (local + remote writers serialize at one NIU).
     * LOCAL_NONATOMIC: plain L1 read-modify-write (NOT atomic; legacy default).
     *
     * @param value The value to increment the semaphore by.
     */
    void up(uint32_t value) {
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            __atomic_add_fetch(reinterpret_cast<uint32_t*>(l1_offset_), value, __ATOMIC_SEQ_CST);
        } else if constexpr (Scope == SemScope::EXTERNAL) {
            noc_semaphore_inc(::get_noc_addr(l1_offset_), value);
            noc_async_atomic_barrier();
        } else {  // LOCAL_NONATOMIC
            *local_ptr() += value;
        }
    }

    /**
     * @brief Atomically increment the semaphore by the specified value on a remote core.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @param value The value to increment the semaphore by.
     * @param vc The virtual channel to use for the transaction (default is NOC_UNICAST_WRITE_VC).
     */
    void up(const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t value, uint8_t vc = NOC_UNICAST_WRITE_VC) {
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "remote up() is not valid on a DM_LOCAL_CACHED semaphore (it would be touched via the NoC); "
            "use SemScope::EXTERNAL");
        const uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, noc.get_noc_id());
        noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
    }

    /**
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     *
     * DM_LOCAL_CACHED: atomic AMO subtract after a coherent spin. As with EXTERNAL, the
     *   check-then-subtract pair is only single-consumer-safe (two consumers can both pass
     *   the >=value spin and both subtract, underflowing); multi-consumer down must be
     *   host-guarded (Phase-2) or use CAS.
     * EXTERNAL: the decrement is an ATOMIC self-targeted NoC RMW (INCR_GET with a negative
     *   increment; wrap=31 => 32-bit modular subtract), so it serializes with concurrent
     *   producer increments at the NIU — no lost update. (HW/emu-verified:
     *   NocAtomicOpsFixture.TestAtomicDecrementIncrGet.) The wait-then-decrement is still
     *   only single-consumer-safe (the >=value check and the decrement are separate); a
     *   multi-consumer EXTERNAL decrement must be host-guarded (Phase-2) or use CAS.
     * LOCAL_NONATOMIC: legacy single-owner (non-atomic) decrement after an uncached spin.
     *
     * @param value The value to decrement the semaphore by.
     */
    void down(uint32_t value) {
        auto* sem_addr = local_ptr();
        WAYPOINT("NSDW");
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            while ((*sem_addr) < value) {  // DM cores are mutually coherent; no invalidate needed
            }
            WAYPOINT("NSDD");
            __atomic_sub_fetch(reinterpret_cast<uint32_t*>(l1_offset_), value, __ATOMIC_SEQ_CST);
        } else if constexpr (Scope == SemScope::EXTERNAL) {
            do {
                invalidate_l1_cache();
            } while ((*sem_addr) < value);
            WAYPOINT("NSDD");
            // Atomic decrement at the NIU: subtract via INCR_GET of the two's-complement of
            // value (wrap=31 gives a full 32-bit modular add). Serializes with concurrent
            // producer increments — unlike the old *sem -= value, which could lose one.
            noc_semaphore_inc(::get_noc_addr(l1_offset_), (uint32_t)(0u - value));
            noc_async_atomic_barrier();
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
    void wait(uint32_t value) { noc_semaphore_wait(local_ptr(), value); }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    void wait_min(uint32_t value) { noc_semaphore_wait_min(local_ptr(), value); }

    /**
     * @brief Set the semaphore to the specified value.
     *
     * @param value The value to set the semaphore to.
     */
    void set(uint32_t value) { noc_semaphore_set(local_ptr(), value); }

    /**
     * @brief Read the current semaphore value through this scope's coherent view.
     */
    uint32_t value() const {
        invalidate_l1_cache();
        return *local_ptr();
    }

    /**
     * @brief Relay this semaphore's local value into a different remote semaphore on a single core.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). To bump the
     *       same semaphore on a remote core, use up(noc, noc_x, noc_y, value) instead.
     *       Writes 4 bytes from this->l1_offset_ to dst_sem.l1_offset_ on the remote core
     *       (noc_x, noc_y).
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param dst_sem The destination Semaphore whose L1 offset receives the value.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @tparam dst_core_type Programmable core type of the destination (defaults to this Semaphore's core_type).
     */
    template <ProgrammableCoreType dst_core_type = core_type, SemScope dst_scope = Scope>
    void relay_unicast(const Noc& noc, const Semaphore<dst_core_type, dst_scope>& dst_sem, uint32_t noc_x, uint32_t noc_y) {
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "relay_unicast is a NoC op and is not valid on a DM_LOCAL_CACHED semaphore; use SemScope::EXTERNAL");
        static_assert(
            dst_scope != SemScope::DM_LOCAL_CACHED,
            "the DESTINATION of relay_unicast must not be a DM_LOCAL_CACHED semaphore: its word lives in the "
            "cached-only pool, so relaying to it would put a NoC write into that pool -- the one thing the "
            "pool exists to exclude. Give the destination SemScope::EXTERNAL.");
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
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "set_multicast is a NoC op and is not valid on a DM_LOCAL_CACHED semaphore; use SemScope::EXTERNAL");
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
    template <NocOptions opts = NocOptions::DEFAULT, ProgrammableCoreType dst_core_type = core_type, SemScope dst_scope = Scope>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope>& dst_sem,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "relay_multicast is a NoC op and is not valid on a DM_LOCAL_CACHED semaphore; use SemScope::EXTERNAL");
        static_assert(
            dst_scope != SemScope::DM_LOCAL_CACHED,
            "the DESTINATION of relay_multicast must not be a DM_LOCAL_CACHED semaphore: its word lives in the "
            "cached-only pool, so relaying to it would put a NoC write into that pool -- the one thing the "
            "pool exists to exclude. Give the destination SemScope::EXTERNAL.");
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
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "inc_multicast is a NoC op and is not valid on a DM_LOCAL_CACHED semaphore; use SemScope::EXTERNAL");
        const uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, noc.get_noc_id());
        noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
    }

private:
    uintptr_t l1_offset_;  // physical L1 offset of the semaphore word (cached-alias address)

    // Local access pointer for reads / non-atomic writes. Uncached alias on Quasar for
    // LOCAL_NONATOMIC and EXTERNAL; cached alias for DM_LOCAL_CACHED.
    volatile tt_l1_ptr uint32_t* local_ptr() const {
        uintptr_t addr = l1_offset_;
#ifdef ARCH_QUASAR
        if constexpr (kUseUncachedLocalView) {
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

// CTAD guide: `Semaphore s(sem::x)` where sem::x is a SemAccessor<Id, Scope> deduces
// Semaphore<TENSIX, Scope> — the host-baked scope — with no explicit template args.
// (Semaphores are TENSIX-scoped; the token carries only Id + SemScope.)
template <uint32_t Id, SemScope S>
Semaphore(SemAccessor<Id, S>) -> Semaphore<ProgrammableCoreType::TENSIX, S>;

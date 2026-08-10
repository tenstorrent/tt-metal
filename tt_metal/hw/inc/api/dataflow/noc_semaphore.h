// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include <hostdevcommon/sem_scope.h>               // enum class SemScope (shared host/device)
#include "api/dataflow/semaphore_binding_token.h"  // SemaphoreBindingToken<Id,Scope,Access> baked-sem token

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The Semaphore class provides a simple interface for semaphore-based synchronization
 * between programmable cores. It allows incrementing and decrementing the semaphore value,
 * as well as waiting for the semaphore to reach a desired value. The semaphore can be
 * manipulated locally or remotely via the NoC.
 *
 * The Scope template parameter selects the physical path / atomicity guarantee — see
 * hostdevcommon/sem_scope.h. It defaults to LOCAL_NONATOMIC (legacy behavior).
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
// Access enforces the host's declared AccessType at compile time: down() needs CONSUME,
// set()/set_multicast() need SET, up()/inc_multicast() anything but OBSERVE; reads are always
// legal. Deduced from the baked token by the CTAD guide after the class. The RAW default keeps
// raw-id call sites compiling unchanged (no declared label to enforce).
template <
    ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX,
    SemScope Scope = SemScope::LOCAL_NONATOMIC,
    SemAccess Access = SemAccess::RAW>
class Semaphore {
    // Lets relay_unicast / relay_multicast read dst_sem's private members. The parameter list
    // must match the class template's arity exactly, or every instantiation is ill-formed.
    template <ProgrammableCoreType OT, SemScope OS, SemAccess OA>
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
    static constexpr bool kUseUncachedLocalView = (Scope != SemScope::DM_LOCAL_CACHED);

    // DM_LOCAL_CACHED is Quasar-DM-only. Host FATALs cover only DECLARED bindings, so the raw-id
    // back door is closed at compile time: these class-scope asserts cover off-Quasar and Quasar
    // TRISC; the raw-id ctor's body assert covers Quasar DM (no binding -> no seeder runs).
#if !defined(ARCH_QUASAR)
    static_assert(
        Scope != SemScope::DM_LOCAL_CACHED,
        "SemScope::DM_LOCAL_CACHED is a Gen2 (Quasar) mechanism -- the cached-only pool, its cache "
        "aliases and its seeding do not exist here. Use SemScope::EXTERNAL.");
#elif defined(COMPILE_FOR_TRISC)
    static_assert(
        Scope != SemScope::DM_LOCAL_CACHED,
        "SemScope::DM_LOCAL_CACHED is reachable only from a data-movement kernel: the pool lives in the "
        "DM cache domain and its seeder is not emitted for TRISC. Use SemScope::EXTERNAL.");
#endif

    // Physical L1 offset of semaphore `id` for THIS scope: DM_LOCAL_CACHED uses the dedicated
    // cached-only pool (MEM_DM_CACHED_SEM_BASE); every other scope uses the normal ring
    // (get_semaphore). MEM_L1_BASE == 0, so the offset is also the cached-alias address.
    static uintptr_t sem_l1_offset(uint32_t id) {
#ifdef ARCH_QUASAR
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            // The token ctor already bounds-checks Id at compile time; runtime defense-in-depth.
            ASSERT(id < MEM_DM_CACHED_SEM_SIZE / L1_ALIGNMENT);
            return static_cast<uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * L1_ALIGNMENT;
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
    explicit Semaphore(uint32_t semaphore_id) : l1_offset_(sem_l1_offset(semaphore_id)) {
        // Fires only when THIS ctor is used: a raw id has no declared binding, so the host never
        // injects the cached-pool seeder -- the word would be boot garbage or a stale count.
        static_assert(
            Scope != SemScope::DM_LOCAL_CACHED,
            "a DM_LOCAL_CACHED semaphore must be constructed from its sem:: binding token "
            "(`Semaphore s(sem::<name>);`): the declared binding is what makes the host emit and "
            "auto-inject the cached-pool seeder. A raw id would address a stale/unseeded pool "
            "word. For an id-addressed semaphore use the default scope or SemScope::EXTERNAL.");
    }

    // Construct from a host-baked sem:: accessor token. Scope and Access come from the token
    // via the CTAD guide after the class, so `Semaphore s(sem::x);` needs no explicit <>.
    // The static_asserts reject an explicit `Semaphore<...>` that mismatches the token.
    template <uint32_t Id, SemScope TokenScope, SemAccess TokenAccess>
    explicit Semaphore(SemaphoreBindingToken<Id, TokenScope, TokenAccess>) : l1_offset_(sem_l1_offset(Id)) {
#ifdef ARCH_QUASAR
        // The pool is indexed by id, so a high id must not run past the pool.
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            static_assert(
                Id * L1_ALIGNMENT < MEM_DM_CACHED_SEM_SIZE,
                "semaphore id does not fit the DM cached semaphore pool (grow MEM_DM_CACHED_SEM_SIZE)");
        }
        // The lock region sits right below the cached pool, so an out-of-range id would CAS-write
        // a live cached counter. (The raw-id path is guarded only by the ASSERT in
        // external_lock_l1_offset().)
        if constexpr (Scope == SemScope::EXTERNAL) {
            static_assert(
                Id * L1_ALIGNMENT < MEM_NOC_SEM_LOCK_SIZE,
                "semaphore id does not fit the EXTERNAL lock region (grow MEM_NOC_SEM_LOCK_SIZE)");
        }
#endif
        static_assert(
            TokenScope == Scope,
            "The sem:: accessor's baked SemScope does not match this Semaphore's Scope. Write "
            "`Semaphore s(sem::x);` and let CTAD pick the baked scope; do not spell "
            "`Semaphore<...>` explicitly on a baked accessor.");
        // Otherwise an explicit `Semaphore<...>` could silently swap the declared access for RAW.
        static_assert(
            TokenAccess == Access,
            "The sem:: accessor's host-declared access (AccessType) does not match this Semaphore's "
            "Access parameter. Write `Semaphore s(sem::x);` and let CTAD adopt it.");
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
        // The access asserts sit in function bodies (not class scope) so they fire only on a
        // mutating CALL -- constructing any binding and wait()/wait_min()/value() stay legal.
        // Never add an explicit instantiation of Semaphore; it would fire them all at once.
        static_assert(
            Access != SemAccess::OBSERVE,
            "up() writes this semaphore, but its host binding is declared AccessType::OBSERVE. "
            "Relabel the binding INCREMENT, or stop writing -- wait()/wait_min()/value() remain "
            "available under OBSERVE.");
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
            Access != SemAccess::OBSERVE,
            "remote up() writes this semaphore on another core, but its host binding is declared "
            "AccessType::OBSERVE. Relabel it INCREMENT.");
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
    void down(uint32_t value) {
        // down() through any label but CONSUME would vacate the host's off-node-consumer
        // rejection, which keys on that label.
        static_assert(
            Access == SemAccess::CONSUME || Access == SemAccess::RAW,
            "down() consumes this semaphore, so its host binding must be declared "
            "AccessType::CONSUME (the off-node-consumer rejection keys on that label). To only "
            "wait on a value without consuming it, use wait_min().");
        auto* sem_addr = local_ptr();
        WAYPOINT("NSDW");
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
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
        } else if constexpr (Scope == SemScope::EXTERNAL) {
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
                noc_fast_atomic_cas4<DM_DEDICATED_NOC, /*program_ret_addr=*/true>(
                    noc_index,
                    write_at_cmd_buf,
                    lock_noc,
                    NOC_UNICAST_WRITE_VC,
                    cmp4,
                    swap4,
                    /*linked=*/false,
                    /*posted=*/false,
                    ret_slot);
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
#else
            // Gen1 single-consumer path: spin, then atomic subtract. (The TRISC leg of the gate
            // is parse-safety only.)
            do {
                invalidate_l1_cache();
            } while ((*sem_addr) < value);
            WAYPOINT("NSDD");
            noc_semaphore_inc(::get_noc_addr(l1_offset_), (uint32_t)(0u - value));
            noc_async_atomic_barrier();
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
    void set(uint32_t value) {
        // The racing-SET rejection keys on the SET label; any other label would vacate it.
        static_assert(
            Access == SemAccess::SET || Access == SemAccess::RAW,
            "set() destructively overwrites this semaphore, so its host binding must be declared "
            "AccessType::SET (the racing-SET rejection keys on that label).");
        noc_semaphore_set(local_ptr(), value);
    }

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
    // DstAccess is its own parameter: relay READS *this and WRITES dst_sem, so any source
    // access is legitimate and only the DESTINATION's label is enforced (a relay is a remote
    // set()). It must be deducible so the user gets the static_assert, not "no matching function".
    template <
        ProgrammableCoreType dst_core_type = core_type,
        SemScope dst_scope = Scope,
        SemAccess DstAccess = SemAccess::RAW>
    void relay_unicast(
        const Noc& noc, const Semaphore<dst_core_type, dst_scope, DstAccess>& dst_sem, uint32_t noc_x, uint32_t noc_y) {
        static_assert(
            DstAccess == SemAccess::SET || DstAccess == SemAccess::RAW,
            "relay_unicast() destructively writes the DESTINATION semaphore (a remote set()), so "
            "the destination binding must be declared AccessType::SET.");
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
        // Writes peer cores' copies rather than this core's own word -- still a destructive set.
        static_assert(
            Access == SemAccess::SET || Access == SemAccess::RAW,
            "set_multicast() destructively writes this semaphore on the destination cores, so its "
            "host binding must be declared AccessType::SET.");
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
    // See relay_unicast: the gate is on the DESTINATION, not on *this.
    template <
        NocOptions opts = NocOptions::DEFAULT,
        ProgrammableCoreType dst_core_type = core_type,
        SemScope dst_scope = Scope,
        SemAccess DstAccess = SemAccess::RAW>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope, DstAccess>& dst_sem,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        static_assert(
            DstAccess == SemAccess::SET || DstAccess == SemAccess::RAW,
            "relay_multicast() destructively writes the DESTINATION semaphore on every core in the "
            "region (a remote set()), so the destination binding must be declared AccessType::SET.");
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
        // Like set_multicast, but INCREMENT-shaped, so any writer label admits it.
        static_assert(
            Access != SemAccess::OBSERVE,
            "inc_multicast() increments this semaphore on the destination cores, but its host binding "
            "is declared AccessType::OBSERVE. Relabel it INCREMENT.");
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

// CTAD guide: `Semaphore s(sem::x)` deduces Semaphore<TENSIX, Scope, Access> from the token
// (semaphores are TENSIX-scoped). A hand-spelled `Semaphore<...>` that mismatches the token trips
// the token ctor's static_asserts; a matching one is caught by test_semaphore_binding_hygiene.
template <uint32_t Id, SemScope S, SemAccess A>
Semaphore(SemaphoreBindingToken<Id, S, A>) -> Semaphore<ProgrammableCoreType::TENSIX, S, A>;

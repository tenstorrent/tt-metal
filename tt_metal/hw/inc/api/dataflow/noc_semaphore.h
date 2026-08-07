// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include <hostdevcommon/sem_scope.h>       // enum class SemScope (shared host/device)
#include "api/dataflow/semaphore_binding_token.h"  // SemaphoreBindingToken<Id,Scope,ReadOnly> baked-sem token

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
// ReadOnly is the host's AccessType::OBSERVE declaration, made enforceable: when true, every mutator
// below fails to compile while wait()/wait_min()/value() keep working. It is deduced from the baked
// token by the CTAD guide after the class, so `Semaphore s(sem::x);` picks it up with no kernel-source
// change. Trailing and defaulted so `Semaphore<>` and `Semaphore<core_type>` keep compiling unchanged.
template <
    ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX,
    SemScope Scope = SemScope::LOCAL_NONATOMIC,
    bool ReadOnly = false>
class Semaphore {
    // Lets relay_unicast / relay_multicast read dst_sem's private members. The parameter list
    // must match the class template's arity exactly, or every instantiation is ill-formed.
    template <ProgrammableCoreType OT, SemScope OS, bool ORO>
    friend class Semaphore;

    // LOCAL_NONATOMIC and EXTERNAL access the local word through the uncached alias on
    // Quasar (so DM reads/writes are coherent with NoC atomics landing at TL1).
    // DM_LOCAL_CACHED uses the cached alias (RISC-V AMOs require the cache and hang on
    // the uncached alias; DM cores are mutually coherent on the cached alias).
    //
    // Cache discipline: each scope keeps its word in exactly one view, so no flush or
    // invalidate is REQUIRED here. The uncached scopes bypass the cache; the cached pool is
    // written only by mutually-coherent DM cores (nothing NoC-writes it), so a flush would
    // publish to nobody and an invalidate could discard a live count. The one mandatory cache
    // op lives in the generated pool seeder (genfiles.cpp), which reads the dispatcher-written
    // ring slot through the UNCACHED alias.
    //
    // LOAD-BEARING: the invalidate_l1_cache() calls below (and in noc_semaphore_wait/wait_min)
    // are a documented NO-OP on Quasar DM cores (tt-2xx/risc_common.h). If that ever becomes a
    // real discard-without-writeback, a cached semaphore would silently lose increments.
    static constexpr bool kUseUncachedLocalView = (Scope != SemScope::DM_LOCAL_CACHED);

    // DM_LOCAL_CACHED is Quasar-DM-only. The host FATALs only cover DECLARED bindings, so these
    // asserts close the raw-id `Semaphore<..., DM_LOCAL_CACHED> s(raw_id)` back door at compile
    // time. Off Quasar the pool does not exist; on Quasar TRISC the pool is in the DM cache
    // domain and the seeder is not emitted, so a TRISC AMO would hit an unseeded word from
    // outside the coherence domain.
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
    // cached-only pool (MEM_DM_CACHED_SEM_BASE, disjoint from the NoC-written kernel_config
    // ring); every other scope uses the normal ring (get_semaphore). MEM_L1_BASE == 0, so the
    // offset is also the cached-alias address.
    static uintptr_t sem_l1_offset(uint32_t id) {
#ifdef ARCH_QUASAR
        if constexpr (Scope == SemScope::DM_LOCAL_CACHED) {
            // Covers the raw-id ctor too; the token ctor also checks this at compile time.
            ASSERT(id * L1_ALIGNMENT < MEM_DM_CACHED_SEM_SIZE);
            return static_cast<uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * L1_ALIGNMENT;
        }
#endif
        return get_semaphore<core_type>(id);
    }

public:
    // l1_offset_ holds the physical L1 offset of the semaphore word (the cached-alias
    // address; MEM_L1_BASE == 0). Local views and NoC addresses are derived from it.
    explicit Semaphore(uint32_t semaphore_id) : l1_offset_(sem_l1_offset(semaphore_id)) {}

    // Construct from a host-baked sem:: accessor token. Scope and ReadOnly come from the
    // token via CTAD (deduction guide after the class), so `Semaphore s(sem::x);` picks the
    // baked mechanism with no explicit <>. The static_asserts reject an explicit
    // `Semaphore<...>` that mismatches the token instead of silently overriding it.
    template <uint32_t Id, SemScope TokenScope, bool TokenReadOnly>
    explicit Semaphore(SemaphoreBindingToken<Id, TokenScope, TokenReadOnly>) : l1_offset_(sem_l1_offset(Id)) {
#ifdef ARCH_QUASAR
        // The pool is indexed by id, so a high id must not run past the pool.
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
        // Without this, an explicit `Semaphore<...>` could match on scope and silently drop the
        // host's OBSERVE restriction via the ReadOnly class default.
        static_assert(
            TokenReadOnly == ReadOnly,
            "The sem:: accessor's host-declared access rights (AccessType::OBSERVE => read-only) do "
            "not match this Semaphore's ReadOnly parameter. Write `Semaphore s(sem::x);` and let "
            "CTAD adopt them.");
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
        // The ReadOnly asserts sit in function bodies (not class scope) so they fire only on a
        // mutating CALL -- constructing an observer and wait()/wait_min()/value() stay legal.
        // Never add an explicit instantiation of Semaphore; it would fire them all at once.
        static_assert(
            !ReadOnly,
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
            !ReadOnly,
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
     * DM_LOCAL_CACHED: atomic AMO subtract after a coherent spin.
     * EXTERNAL: atomic self-targeted NoC RMW (INCR_GET with a negative increment; wrap=31 =>
     *   32-bit modular subtract), so it serializes with concurrent producer increments at
     *   the NIU — no lost update.
     * LOCAL_NONATOMIC: legacy single-owner (non-atomic) decrement after an uncached spin.
     *
     * All scopes: the >=value spin and the decrement are separate, so down() is only
     * single-consumer-safe; a multi-consumer down must be host-guarded or use CAS.
     *
     * @param value The value to decrement the semaphore by.
     */
    void down(uint32_t value) {
        // The >=value spin is a read, but the subtract that follows is not: down() is a writer.
        static_assert(
            !ReadOnly,
            "down() decrements this semaphore, but its host binding is declared AccessType::OBSERVE. "
            "Relabel it CONSUME. To only wait on a value without consuming it, use wait_min().");
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
            // Atomic subtract at the NIU: INCR_GET of the two's complement (wrap=31 => full
            // 32-bit modular add); serializes with concurrent producer increments.
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
    void set(uint32_t value) {
        static_assert(
            !ReadOnly,
            "set() destructively overwrites this semaphore, but its host binding is declared "
            "AccessType::OBSERVE. Relabel it SET.");
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
    // DstReadOnly is its own parameter: relay READS *this and WRITES dst_sem, so a read-only
    // SOURCE is legitimate and only a read-only DESTINATION is the violation. It must be
    // deducible so the user gets the static_assert, not "no matching function".
    template <ProgrammableCoreType dst_core_type = core_type, SemScope dst_scope = Scope, bool DstReadOnly = false>
    void relay_unicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope, DstReadOnly>& dst_sem,
        uint32_t noc_x,
        uint32_t noc_y) {
        static_assert(
            !DstReadOnly,
            "relay_unicast() writes the DESTINATION semaphore, whose host binding is declared "
            "AccessType::OBSERVE. Relabel the destination binding SET (a read-only source is fine).");
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
        // Writes peer cores' copies rather than this core's own word -- still a write.
        static_assert(
            !ReadOnly,
            "set_multicast() writes this semaphore on the destination cores, but its host binding is "
            "declared AccessType::OBSERVE. Relabel it SET.");
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
        bool DstReadOnly = false>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope, DstReadOnly>& dst_sem,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        static_assert(
            !DstReadOnly,
            "relay_multicast() writes the DESTINATION semaphore on every core in the region, but that "
            "binding is declared AccessType::OBSERVE. Relabel the destination binding SET (a "
            "read-only source is fine).");
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
        // Same reasoning as set_multicast: writes peer cores' copies, still a write.
        static_assert(
            !ReadOnly,
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

// CTAD guide: `Semaphore s(sem::x)` where sem::x is a SemaphoreBindingToken<Id, Scope, ReadOnly>
// deduces Semaphore<TENSIX, Scope, ReadOnly> — the host-baked mechanism AND the host-declared access
// rights — with no explicit template args. (Semaphores are TENSIX-scoped.)
// A hand-spelled `Semaphore<...>` that mismatches the token trips the two static_asserts in the
// token ctor; one that happens to match compiles and is caught by the hygiene sweep
// (test_semaphore_binding_hygiene) instead.
template <uint32_t Id, SemScope S, bool RO>
Semaphore(SemaphoreBindingToken<Id, S, RO>) -> Semaphore<ProgrammableCoreType::TENSIX, S, RO>;

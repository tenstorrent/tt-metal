// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"

// Force-inlining is what makes the host-chosen mechanism constant-fold away on Quasar (a cached
// up() becomes a single bare AMO). Gen1 resolves every semaphore to LOCAL_NONATOMIC anyway, so
// there the attribute only perturbs the compiler's own inlining choices -- leave those to it.
#if defined(ARCH_QUASAR)
#define TT_SEM_INLINE __attribute__((always_inline))
#else
#define TT_SEM_INLINE
#endif

/**
 * @brief Physical path a semaphore's accesses take. This is picked by the host at compile time
 *        and baked into the kernel. The path picked provides the fastest access that keeps the
 *        semaphore operations atomic.
 *
 *  - LOCAL_NONATOMIC: Stored in L1 and accessed by read-modify-write. Picked only when at most
 *                     one binder instance exists.
 *  - DM_LOCAL_CACHED: Stored in a dedicated L1 pool and accessed through the DM cache via
 *                     RISC-V AMO. Picked only when all binders are DMs on the same node where
 *                     the semaphore exists.
 *  - EXTERNAL:        Stored in L1 and accessed through atomics operations via the NOC. Picked
 *                     whenever the semaphore is reachable beyond a single node. An EXTERNAL
 *                     semaphore's value can never be 0xFFFFFFFF, it would look like a NoC
 *                     atomic's reply that has not yet arrived.
 *
 * Per-kernel scope tables are constructed at build time by the host and baked into the kernel.
 * The kernel code can then query the scope of a semaphore for the appropriate mechanism to use.
 *
 * @note Never access a bound semaphore's word directly (get_semaphore(), the noc_semaphore_*
 *       free functions, raw pointers), always go through this class. A raw access is
 *       invisible to the host's mechanism choice and can silently race it.
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
};

// Looks up the host-chosen mechanism for a semaphore id. Codegen injects the table
// (TT_METAL2_SEM_SCOPE_TABLE) into each Metal 2.0 kernel at build time, any id without
// a table entry uses LOCAL_NONATOMIC.
inline TT_SEM_INLINE constexpr SemScope sem_scope_of(uint32_t semaphore_id) {
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
 * manipulated locally or remotely via the NoC. The physical mechanism is host-chosen.
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
 *  - value(): Read the current semaphore value.
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

    // DM_LOCAL_CACHED semaphores live in their own dedicated pool
    static TT_SEM_INLINE inline uintptr_t sem_l1_offset(uint32_t id) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        if (sem_scope_of(id) == SemScope::DM_LOCAL_CACHED) {
            ASSERT(id < MEM_DM_CACHED_SEM_SIZE / MEM_DM_CACHED_SEM_ROW);
            return static_cast<uintptr_t>(MEM_DM_CACHED_SEM_BASE) + id * MEM_DM_CACHED_SEM_ROW;
        }
#endif
        return get_semaphore<core_type>(id);
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC) && !defined(TT_EMULE_USE_L1_POOL)
    // This EXTERNAL semaphore's lock word on this node. Locks are spaced one per 16B row so
    // the NoC-CAS always targets the first 4-byte word.
    uint32_t external_lock_l1_offset() const {
        const uint32_t id =
            (static_cast<uint32_t>(l1_offset_) - static_cast<uint32_t>(get_semaphore<core_type>(0))) / L1_ALIGNMENT;
        ASSERT(id * L1_ALIGNMENT < MEM_NOC_SEM_LOCK_SIZE);
        return MEM_NOC_SEM_LOCK_BASE + id * L1_ALIGNMENT;
    }
    // This hart's private CAS-return slot.
    static uint32_t cas_ret_slot() {
        uint64_t hart;
        asm volatile("csrr %0, mhartid" : "=r"(hart));
        ASSERT(static_cast<uint32_t>(hart) * 4 < MEM_NOC_CAS_RET_SIZE);
        return MEM_NOC_CAS_RET_BASE + static_cast<uint32_t>(hart) * 4;
    }
#endif

public:
    // l1_offset_ holds the physical L1 offset of the semaphore word.
    explicit TT_SEM_INLINE Semaphore(uint32_t semaphore_id) :
        l1_offset_(sem_l1_offset(semaphore_id))
#if defined(ARCH_QUASAR)
        ,
        scope_(sem_scope_of(semaphore_id))
#endif
    {
        // Gen1 hardcodes scope_ above; trip if the host ever bakes anything else.
        ASSERT(sem_scope_of(semaphore_id) == scope_);
    }

    /**
     * @brief Increment the semaphore by the specified value.
     * @note Currently atomicity is not guaranteed on WH/BH, multiple cores incrementing simultaneously may lead to lost
     * updates.
     *
     * DM_LOCAL_CACHED: atomic 32-bit AMO on the cached alias.
     * EXTERNAL:        self-targeted NoC atomic increment.
     * LOCAL_NONATOMIC: L1 read-modify-write (not atomic).
     *
     * @param value The value to increment the semaphore by.
     */
    TT_SEM_INLINE void up(uint32_t value) {
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
            ASSERT(false);  // compute kernels cannot bind semaphores
#endif
        } else {  // LOCAL_NONATOMIC
            *local_ptr() += value;
        }
    }

    /**
     * @brief Atomically increment the semaphore by the specified value on a remote core.
     *
     * On a DM_LOCAL_CACHED semaphore the only legal target is this node (all its binders are
     * on-node -- that is why the census picked cached), so the increment is served by the local
     * AMO. This keeps the portable pattern up(noc, my_x, my_y, v) working under every scope.
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @param value The value to increment the semaphore by.
     * @param vc The virtual channel to use for the transaction (default is NOC_UNICAST_WRITE_VC).
     */
    TT_SEM_INLINE void up(
        const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t value, uint8_t vc = NOC_UNICAST_WRITE_VC) {
        if (scope_ == SemScope::DM_LOCAL_CACHED) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
            // A NoC atomic must never touch the cached pool; instead use an AMO.
            ASSERT(noc.is_local_bank(noc_x, noc_y));
            up(value);
#else
            ASSERT(false);  // the host never bakes CACHED for this platform
#endif
            return;
        }
        const uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, noc.get_noc_id());
        noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
    }

    /**
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     * @note Currently atomicity is not guaranteed on WH/BH, multiple cores incrementing simultaneously may lead to lost
     * updates.
     *
     * DM_LOCAL_CACHED: multi-consumer-safe via LR/SC retry loop.
     * EXTERNAL:        multi-consumer-safe via a NoC-CAS lock, consumers must run on the semaphore's node.
     * LOCAL_NONATOMIC: single-owner (non-atomic) decrement.
     *
     * @param value The value to decrement the semaphore by.
     */
    TT_SEM_INLINE void down(uint32_t value) {
        auto* sem_addr = local_ptr();
        WAYPOINT("NSDW");
        if (scope_ == SemScope::DM_LOCAL_CACHED) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
            auto* word = reinterpret_cast<uint32_t*>(l1_offset_);  // cached alias
            uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
            do {
                WAYPOINT("NSDW");
                while (observed < value) {
                    observed = __atomic_load_n(word, __ATOMIC_RELAXED);
                }
                WAYPOINT("NSDD");
            } while (!__atomic_compare_exchange_n(
                word, &observed, observed - value, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
#else
            ASSERT(false);  // the host census never bakes CACHED for this platform
#endif
        } else if (scope_ == SemScope::EXTERNAL) {
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC) && !defined(TT_EMULE_USE_L1_POOL)
            // Only consumers need to lock; producers can NoC-increment without contention.
            noc_async_atomic_barrier();  // Wait until all prior NoC atomics have completed.
            const uint64_t sem_noc = ::get_noc_addr(l1_offset_);
            const uint64_t lock_noc = ::get_noc_addr(external_lock_l1_offset());
            const uint32_t ret_slot = cas_ret_slot();
            auto* ret_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) + ret_slot);
            constexpr uint32_t kCasSentinel = 0xFFFFFFFFu;
            auto consumed_cas4 = [&](uint32_t cmp4, uint32_t swap4) -> uint32_t {
                *ret_word = kCasSentinel;
                noc_fast_atomic_cas4<DM_DEDICATED_NOC>(
                    noc_index, lock_noc, NOC_UNICAST_WRITE_VC, cmp4, swap4, ret_slot);
                noc_async_atomic_barrier();
                WAYPOINT("NSDD");
                while (*ret_word == kCasSentinel) {
                }
                return *ret_word;
            };
            for (;;) {
                // Wait until the semaphore has enough credit(s)
                WAYPOINT("NSDW");
                do {
                    invalidate_l1_cache();
                } while ((*sem_addr) < value);
                if (consumed_cas4(/*cmp4=*/0, /*swap4=*/1) != 0) {
                    continue;  // another consumer holds the lock, wait for it to release
                }
                // Re-check under the lock, no other consumer can decrement now.
                invalidate_l1_cache();
                const bool ok = (*sem_addr) >= value;
                if (ok) {
                    WAYPOINT("NSDD");
                    // Atomic subtract: the NoC only has atomic ADD (INCR_GET), so add the two's complement.
                    noc_semaphore_inc(sem_noc, (uint32_t)(0u - value));
                    noc_async_atomic_barrier();
                    WAYPOINT("NSDD");
                    // confirm the subtract's reply was received
                    for (uint32_t spin = 0; spin < 1024u && *ret_word == kCasSentinel; spin++) {
                    }
                }
                consumed_cas4(/*cmp4=*/1, /*swap4=*/0);  // release
                if (ok) {
                    return;
                }
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
            ASSERT(false);  // compute kernels cannot bind semaphores.
#endif
        } else {  // LOCAL_NONATOMIC
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
    TT_SEM_INLINE void wait(uint32_t value) { noc_semaphore_wait(local_ptr(), value); }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    TT_SEM_INLINE void wait_min(uint32_t value) { noc_semaphore_wait_min(local_ptr(), value); }

    /**
     * @brief Set the semaphore to the specified value.
     *
     * @note A non-atomic destructive store under every scope.
     *
     * @param value The value to set the semaphore to.
     */
    TT_SEM_INLINE void set(uint32_t value) { noc_semaphore_set(local_ptr(), value); }

    /**
     * @brief Read the current semaphore value through this scope's coherent view.
     */
    TT_SEM_INLINE uint32_t value() const {
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
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);
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
        ASSERT(scope_ != SemScope::DM_LOCAL_CACHED);
        const uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, noc.get_noc_id());
        noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
    }

private:
    uintptr_t l1_offset_;  // physical L1 offset of the semaphore word (cached-alias address)
#if defined(ARCH_QUASAR)
    SemScope scope_;  // host-chosen mechanism (from the codegen table)
#else
    // Gen1: ResolveSemaphoreScope is Gen2-gated and always resolves LOCAL_NONATOMIC, so the
    // mechanism is a compile-time constant here. Keeping it out of the object leaves the class
    // exactly the size and shape it has always been, so Gen1 codegen is untouched.
    static constexpr SemScope scope_ = SemScope::LOCAL_NONATOMIC;
#endif

    // Local access pointer for reads / non-atomic writes.
    TT_SEM_INLINE volatile tt_l1_ptr uint32_t* local_ptr() const {
        uintptr_t addr = l1_offset_;
#ifdef ARCH_QUASAR
        if (scope_ != SemScope::DM_LOCAL_CACHED) {
            addr += MEM_L1_UNCACHED_BASE;
        }
#endif
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    // Physical L1 offset used to form NoC addresses.
    uintptr_t get_l1_addr() const { return l1_offset_; }

    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint8_t noc) const {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, get_l1_addr(), noc);
    }

    uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, get_l1_addr(), noc);
    }
};

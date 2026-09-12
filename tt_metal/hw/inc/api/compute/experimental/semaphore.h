// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/semaphore_binding_token.h"  // SemScope + SemaphoreBindingToken

#ifdef COMPILE_FOR_TRISC
#include "api/compute/experimental/semaphore_compute_impl.h"
#else
#include "api/dataflow/semaphore_dm_impl.h"
#endif

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The host picks the access mechanism (SemScope) from where the semaphore's binder kernels run and
 * delivers it in a generated binding token; construct with CTAD, `Semaphore s(sem::name);`.
 *
 * DM builds expose local operations plus the NoC operations (remote up, set/relay/inc multicast).
 *
 * Blackhole UNPACK/PACK builds expose the local operations only, on the Tensix hardware (Sync Unit)
 * semaphore, and require SemScope::COMPUTE_ATOMIC; any other scope is a compile error, so a compute
 * kernel cannot reach a semaphore through a non-atomic path. Compute rules a kernel author must know:
 *  - The value is 0..15. More than 15 outstanding credits lose posts silently; a producer that gates
 *    each up() with wait_not_full() can never get there.
 *  - Its capacity (hardware Max) is SemaphoreAdvancedOptions::max_value, the ring depth in credits;
 *    wait_not_full() blocks while the value is at capacity. Default capacity is 15.
 *  - It starts every kernel at 0 and the kernel must leave it balanced at 0. The 2.0
 *    compute_kernel_hw_startup (LLKOperand overloads) seeds it on PACK; a kernel that does not call that
 *    startup must call set(0) on its producing thread before the first up(). The host rejects a nonzero
 *    initial_value and a second compute semaphore.
 *  - up()/down() are ordered after this thread's engine work (packer/unpacker), and wait()/wait_min()
 *    order this thread's *subsequent engine instructions* after the condition, not RISC loads. A
 *    kernel that reads the buffer with the RISC must poll value() instead.
 *  - MATH and isolate-SFPU builds expose no synchronization methods.
 * Details and instruction sequences: semaphore_compute_impl.h.
 */
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX, SemScope SCOPE = SemScope::LOCAL_NONATOMIC>
class Semaphore {
    template <ProgrammableCoreType, SemScope>
    friend class Semaphore;

public:
    template <std::uint32_t SEM_ID, SemScope TOK_SCOPE>
    explicit __attribute__((always_inline)) Semaphore(SemaphoreBindingToken<SEM_ID, TOK_SCOPE>) :
        handle_(semaphore_detail::sem_l1_offset<core_type, SCOPE>(SEM_ID)) {
        static_assert(
            TOK_SCOPE == SCOPE,
            "construct a bound semaphore with CTAD: `Semaphore s(sem::name);`; spelling "
            "`Semaphore<>` fixes the scope to LOCAL_NONATOMIC");
    }

    explicit __attribute__((always_inline)) Semaphore(std::uint32_t semaphore_id) :
        handle_(semaphore_detail::sem_l1_offset<core_type, SCOPE>(semaphore_id)) {
#ifdef COMPILE_FOR_TRISC
        // A runtime id is invisible to the host semaphore census, so the host cannot know a
        // compute thread touches this word and cannot force every participant onto the same
        // mechanism -- a DM kernel binding the same word would resolve to LOCAL_NONATOMIC and its
        // plain read-modify-write would drop this thread's atomic update. Compute therefore takes
        // host-generated binding tokens only.
        static_assert(
            semaphore_detail::always_false<SCOPE>,
            "a compute semaphore cannot be built from a runtime semaphore id: bind it in the "
            "ProgramSpec and construct it from the generated token instead -- `Semaphore "
            "s(sem::name);`");
#else
        static_assert(
            SCOPE == SemScope::LOCAL_NONATOMIC,
            "a runtime semaphore id has no host-resolved mechanism and is LOCAL_NONATOMIC only");
#endif
    }

#if !defined(COMPILE_FOR_TRISC) || (defined(ARCH_BLACKHOLE) && (defined(TRISC_UNPACK) || defined(TRISC_PACK)))
    /**
     * @brief Increment the semaphore by `value`. Never blocks the caller.
     *
     * DM_LOCAL_CACHED: atomic 32-bit AMO on the cached alias.
     * EXTERNAL:        self-targeted NoC atomic increment.
     * LOCAL_NONATOMIC: L1 read-modify-write (not atomic; the host picks it only for a single binder).
     * COMPUTE_ATOMIC:  `value` SEMPOSTs, each ordered after this thread's packer/unpacker work, so a
     *                  consumer that sees the credit also sees the data (publish-after-data). Value +
     *                  outstanding credits must stay <= 15; gate with wait_not_full() to guarantee it.
     *
     * @param value The value to increment the semaphore by.
     */
    __attribute__((always_inline)) void up(std::uint32_t value) {
        semaphore_detail::up<core_type, SCOPE>(handle_, value);
    }

    /**
     * @brief Decrement the semaphore by `value`.
     *
     * DM scopes block until the semaphore is sufficient, then subtract:
     * DM_LOCAL_CACHED: multi-consumer-safe via LR/SC retry loop.
     * EXTERNAL:        multi-consumer-safe via a NoC-CAS lock; consumers must run on the semaphore's node.
     * LOCAL_NONATOMIC: single-owner (non-atomic) decrement.
     * COMPUTE_ATOMIC:  `value` SEMGETs ordered after this thread's engine work (release-after-read). Does
     *                  NOT wait for sufficiency (SEMGET floors at 0): the wait belongs before the engine
     *                  reads the slot and the decrement after, so pair it with wait_min() --
     *                  `wait_min(n); <engine reads slot>; down(n)`.
     *
     * @param value The value to decrement the semaphore by.
     */
    __attribute__((always_inline)) void down(std::uint32_t value) {
        semaphore_detail::down<core_type, SCOPE>(handle_, value);
    }

    /**
     * @brief Block until the semaphore equals `value`. Does not modify it.
     *
     * DM: RISC poll. COMPUTE_ATOMIC: retires this thread's own posted up()/down() first, then RISC-polls.
     *
     * @param value The value to wait for.
     */
    __attribute__((always_inline)) void wait(std::uint32_t value) const {
        semaphore_detail::wait<SCOPE>(handle_, value);
    }

    /**
     * @brief Block until the semaphore is at least `value`. Does not modify it.
     *
     * DM: RISC poll. COMPUTE_ATOMIC with value == 1: Tensix-side SEMWAIT -- the RISC returns at once
     * and this thread's next engine instructions (UNPACR/PACR) are held until the value is nonzero;
     * the fastest form. Other values: as wait().
     *
     * @param value The minimum value to wait for.
     */
    __attribute__((always_inline)) void wait_min(std::uint32_t value) const {
        semaphore_detail::wait_min<SCOPE>(handle_, value);
    }

#ifdef COMPILE_FOR_TRISC
    /**
     * @brief Producer back-pressure (compute only). Block this thread's next engine instructions
     * (PACR/UNPACR) while the semaphore is at its capacity, SemaphoreAdvancedOptions::max_value. The RISC
     * returns at once. Canonical producer loop: `wait_not_full(); pack_tile(ring, slot); up(1);`.
     */
    __attribute__((always_inline)) void wait_not_full() const { semaphore_detail::wait_not_full<SCOPE>(handle_); }
#endif

    /**
     * @brief Set the semaphore to `value`.
     *
     * @note A non-atomic destructive store under every scope; requires a quiescent protocol.
     * DM: plain store. COMPUTE_ATOMIC: SEMINIT (Max = capacity) ordered after this thread's engine work;
     * `value` <= 15.
     *
     * @param value The value to set the semaphore to.
     */
    __attribute__((always_inline)) void set(std::uint32_t value) { semaphore_detail::set<SCOPE>(handle_, value); }

    /**
     * @brief The settled current value.
     *
     * DM: a fresh (cache-invalidated) read; a RISC-side write has already retired. COMPUTE_ATOMIC: first
     * retires this thread's own posted SEMPOST/SEMGET (tensix_sync), then reads the Sync Unit.
     *
     * @return Current semaphore value.
     */
    __attribute__((always_inline)) std::uint32_t value() const {
        return semaphore_detail::current<SCOPE>(handle_);
    }
#endif

#ifndef COMPILE_FOR_TRISC
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
    __attribute__((always_inline)) void up(
        const Noc& noc,
        std::uint32_t noc_x,
        std::uint32_t noc_y,
        std::uint32_t value,
        std::uint8_t vc = NOC_UNICAST_WRITE_VC) {
        semaphore_detail::up_remote<core_type, SCOPE>(handle_, noc, noc_x, noc_y, value, vc);
    }

    /**
     * @brief Relay this semaphore's local value into a different remote semaphore on a single core.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). To bump the
     *       same semaphore on a remote core, use up(noc, noc_x, noc_y, value) instead.
     *       Writes 4 bytes from this semaphore's word to dst_sem's word on the remote core (noc_x, noc_y).
     *
     * @param noc The Noc object representing the NoC to use for the transaction.
     * @param dst_sem The destination Semaphore whose L1 offset receives the value.
     * @param noc_x The X coordinate of the remote core in the NoC.
     * @param noc_y The Y coordinate of the remote core in the NoC.
     * @tparam dst_core_type Programmable core type of the destination (defaults to this Semaphore's core_type).
     */
    template <ProgrammableCoreType dst_core_type = core_type, SemScope dst_scope = SemScope::LOCAL_NONATOMIC>
    void relay_unicast(
        const Noc& noc, const Semaphore<dst_core_type, dst_scope>& dst_sem, std::uint32_t noc_x, std::uint32_t noc_y) {
        semaphore_detail::relay_unicast<SCOPE, dst_scope>(handle_, dst_sem.handle_, noc, noc_x, noc_y);
    }

    /**
     * @brief Set the semaphore value on multiple cores in a specified rectangular region of the NoC.
     * @note Sender cannot be part of the multicast destinations unless opts includes MCAST_INCL_SRC.
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
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t num_dests,
        bool linked = false) {
        semaphore_detail::set_multicast<opts, SCOPE>(
            handle_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, num_dests, linked);
    }

    /**
     * @brief Relay this semaphore's local value into a different destination semaphore on a rectangular region.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). Each core in the region
     *       receives the 4-byte write at dst_sem's L1 offset.
     * @note Sender cannot be part of the multicast destinations unless opts includes MCAST_INCL_SRC.
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
    template <
        NocOptions opts = NocOptions::DEFAULT,
        ProgrammableCoreType dst_core_type = core_type,
        SemScope dst_scope = SemScope::LOCAL_NONATOMIC>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope>& dst_sem,
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t num_dests,
        bool linked = false) {
        semaphore_detail::relay_multicast<opts, SCOPE, dst_scope>(
            handle_, dst_sem.handle_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, num_dests, linked);
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
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t value,
        std::uint32_t num_dests) {
        semaphore_detail::inc_multicast<SCOPE>(
            handle_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, value, num_dests);
    }
#endif

private:
    // DM: the semaphore word's L1 offset. Compute: the Tensix hardware semaphore index (see
    // semaphore_compute_impl.h::sem_l1_offset).
    std::uintptr_t handle_;
};

template <std::uint32_t SEM_ID, SemScope TOK_SCOPE>
Semaphore(SemaphoreBindingToken<SEM_ID, TOK_SCOPE>) -> Semaphore<ProgrammableCoreType::TENSIX, TOK_SCOPE>;

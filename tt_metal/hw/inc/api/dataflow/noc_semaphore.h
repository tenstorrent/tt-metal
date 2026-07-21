// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The Semaphore class provides a simple interface for semaphore-based synchronization
 * between programmable cores. It allows incrementing and decrementing the semaphore value,
 * as well as waiting for the semaphore to reach a desired value. The semaphore can be
 * manipulated locally or remotely via the NoC.
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

public:
    explicit Semaphore(uint32_t semaphore_id) : local_l1_addr_(get_semaphore<core_type>(semaphore_id)) {
#ifdef ARCH_QUASAR
        local_l1_addr_ += MEM_L1_UNCACHED_BASE;
#endif
    }

    /**
     * @brief Increment the semaphore by the specified value.
     * @note Currently atomicity is not guaranteed, multiple cores incrementing simultaneously may lead to lost updates.
     *
     * @param value The value to increment the semaphore by.
     */
    void up(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        *sem_addr += value;
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
        const uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, noc.get_noc_id());
        noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id(), vc);
    }

    /**
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     * @note Currently atomicity is not guaranteed, multiple cores decrementing simultaneously may lead to lost updates.
     *
     * @param value The value to decrement the semaphore by.
     */
    void down(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        WAYPOINT("NSDW");
        do {
            invalidate_l1_cache();
        } while ((*sem_addr) < value);
        WAYPOINT("NSDD");
        *sem_addr -= value;
    }

    // The following methods provide parity with existing semaphore API, but have non-standard semantics.

    /**
     * @brief Block until the semaphore is set to the specified value.
     *
     * @param value The value to wait for.
     */
    void wait(uint32_t value) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    void wait_min(uint32_t value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    /**
     * @brief Set the semaphore to the specified value.
     *
     * @param value The value to set the semaphore to.
     */
    void set(uint32_t value) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    /**
     * @brief Relay this semaphore's local value into a different remote semaphore on a single core.
     * @note dst_sem must be a different Semaphore than this one (a different L1 offset). To bump the
     *       same semaphore on a remote core, use up(noc, noc_x, noc_y, value) instead.
     *       Writes 4 bytes from this->local_l1_addr_ to dst_sem.local_l1_addr_ on the remote core
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
        ASSERT(local_l1_addr_ != dst_sem.local_l1_addr_);
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
        ASSERT(local_l1_addr_ != dst_sem.local_l1_addr_);
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
        const uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, noc.get_noc_id());
        noc_semaphore_inc_multicast(multicast_addr, value, num_dests, noc.get_noc_id());
    }

private:
    uintptr_t local_l1_addr_;

    // L1 offset of this semaphore as a NoC-addressable address (strips the QUASAR uncached alias).
    uintptr_t get_l1_addr() const {
#ifdef ARCH_QUASAR
        return local_l1_addr_ - MEM_L1_UNCACHED_BASE;
#else
        return local_l1_addr_;
#endif
    }

    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint8_t noc) const {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, get_l1_addr(), noc);
    }

    uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, get_l1_addr(), noc);
    }
};

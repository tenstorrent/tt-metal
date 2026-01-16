// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/noc.h"

namespace experimental {

/**
 * @brief Experimental semaphore synchronization primitive for programmable cores.
 *
 * @note This API is experimental and subject to change.
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
 */
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) : local_l1_addr_(get_semaphore<core_type>(semaphore_id)) {}

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
        uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, local_l1_addr_);
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
     * @tparam mcast_mode Indicates whether to include the sender in the multicast (default is EXCLUDE_SRC)
     */
    template <Noc::McastMode mcast_mode = Noc::McastMode::EXCLUDE_SRC>
    void set_multicast(
        const Noc& noc,
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t num_dests,
        bool linked = false) {
        uint64_t multicast_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, local_l1_addr_, noc.get_noc_id());
        if constexpr (mcast_mode == Noc::McastMode::INCLUDE_SRC) {
            noc_semaphore_set_multicast_loopback_src(
                local_l1_addr_, multicast_addr, num_dests, linked, noc.get_noc_id());
        } else if constexpr (mcast_mode == Noc::McastMode::EXCLUDE_SRC) {
            noc_semaphore_set_multicast(local_l1_addr_, multicast_addr, num_dests, linked, noc.get_noc_id());
        }
    }

private:
    uint32_t local_l1_addr_;
};

}  // namespace experimental

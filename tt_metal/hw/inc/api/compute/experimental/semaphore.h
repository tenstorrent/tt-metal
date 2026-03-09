// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
#include "core_config.h"
#include "noc/noc_parameters.h"

namespace ckernel {
#ifdef ARCH_QUASAR
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
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) :
        local_l1_addr_(
            MEM_L1_UNCACHED_BASE +
            (uintptr_t)(sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)] + semaphore_id * L1_ALIGNMENT)) {
        DPRINT << "Semaphore ID: " << semaphore_id << ENDL();
        DPRINT << "Semaphore base address: " << sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)] << ENDL();
        DPRINT << "L1 alignment: " << L1_ALIGNMENT << ENDL();
        DPRINT << "Semaphore address: " << local_l1_addr_ << ENDL();
    }

    /**
     * @brief Increment the semaphore by the specified value.
     * @note Currently atomicity is not guaranteed, multiple cores incrementing simultaneously may lead to lost updates.
     *
     * @param value The value to increment the semaphore by.
     */
    void up(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        DPRINT << "Incrementing semaphore by " << value << ENDL();
        *sem_addr += value;
        DPRINT << "Semaphore incremented to " << *sem_addr << ENDL();
    }

    /**
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     * @note Currently atomicity is not guaranteed, multiple cores decrementing simultaneously may lead to lost updates.
     *
     * @param value The value to decrement the semaphore by.
     */
    void down(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        WAYPOINT("TSDW");
        do {
            // invalidate_l1_cache();
        } while ((*sem_addr) < value);
        WAYPOINT("TSDD");
        *sem_addr -= value;
    }

    // The following methods provide parity with existing semaphore API, but have non-standard semantics.

    /**
     * @brief Block until the semaphore is set to the specified value.
     *
     * @param value The value to wait for.
     */
    void wait(uint32_t value) {
        WAYPOINT("TSWW");
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        DPRINT << "Waiting for semaphore to be set to " << value << ENDL();
        while ((*sem_addr) != value) {
            DPRINT << "Current semaphore value: " << *sem_addr << ENDL();
        }
        DPRINT << "Semaphore set to " << *sem_addr << ENDL();
        WAYPOINT("TSWD");
    }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    void wait_min(uint32_t value) {
        WAYPOINT("TSMWW");
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        do {
            // invalidate_l1_cache();
        } while ((*sem_addr) < value);
        WAYPOINT("TSMWD");
    }

    /**
     * @brief Set the semaphore to the specified value.
     *
     * @param value The value to set the semaphore to.
     */
    void set(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        *sem_addr = value;
    }

private:
    uint32_t local_l1_addr_;
};
#endif
}  // namespace ckernel

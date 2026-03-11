// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "api/compute/common.h"
#include "core_config.h"
#include "noc/noc_parameters.h"

namespace ckernel {
#ifdef ARCH_QUASAR
/**
 * @brief Experimental semaphore synchronization primitive for Tensix engines.
 *
 * @note This API is experimental and subject to change.
 *
 * The Semaphore class provides a simple interface for semaphore-based synchronization
 * between Tensix engines. It allows incrementing and decrementing the semaphore value,
 * as well as waiting for the semaphore to reach a desired value.
 *
 * Usage:
 *   - Construct a Semaphore with a given semaphore ID.
 *   - Use up(), down(), and other methods to perform synchronization.
 *
 * Methods:
 *  - up(value): Increment the semaphore by the specified value locally.
 *  - down(value): Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
 *
 * The following methods (non-standard semantics) are also available, for parity with existing API:
 *  - wait(value): Block until the semaphore is set to the specified value.  Does not decrement the semaphore.
 *  - wait_min(value): Block until the semaphore is at least the specified value.  Does not decrement the semaphore.
 *  - set(value): Set the semaphore to the specified value.
 */
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) :
        local_l1_addr_(
            MEM_L1_UNCACHED_BASE +
            ((uintptr_t)sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)] + semaphore_id * L1_ALIGNMENT)) {}

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
     * @brief Decrement the semaphore by the specified value, blocking until the semaphore is sufficient.
     * @note Currently atomicity is not guaranteed, multiple cores decrementing simultaneously may lead to lost updates.
     *
     * @param value The value to decrement the semaphore by.
     */
    void down(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        WAYPOINT("TSDW");
        while ((*sem_addr) < value);
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
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        WAYPOINT("TSWW");
        while ((*sem_addr) != value);
        WAYPOINT("TSWD");
    }

    /**
     * @brief Block until the semaphore is at least the specified value.
     *
     * @param value The minimum value to wait for.
     */
    void wait_min(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        WAYPOINT("TSWMW");
        while ((*sem_addr) < value);
        WAYPOINT("TSWMD");
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

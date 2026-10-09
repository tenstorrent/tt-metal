// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/kernel_thread_globals.h"
#include "ckernel.h"

#if defined(ARCH_QUASAR) && (defined(TRISC_UNPACK) || defined(TRISC_PACK))
namespace ckernel {

/**
 * @brief Experimental semaphore shared by the 4 NEOs of a Quasar cluster, for handing L1 data from one
 * NEO's packer or unpacker to another NEO's, e.g. PACK on NEO 0 -> UNPACK on NEO 1.
 *
 * @note This API is experimental and subject to change.
 *
 * Backed by cluster semaphore register `index`, a 16-bit counter. The kernel picks the index
 * (0..TENSIX_GLOBAL_SEM_USER_COUNT - 1; 27-31 are reserved); the host does not allocate it.
 *
 *  - One producer thread and one consumer thread per semaphore. down() checks the count and then
 *    subtracts in two steps, so a second consumer could take it below 0, which is fatal on TRISC.
 *  - Firmware sets every one of these semaphores to 0 at boot and nothing resets them between kernels,
 *    so a kernel must leave each semaphore it uses at 0. A prefilled ring must initialize its count
 *    before any participant uses it, and reset it after all participants finish.
 *  - UNPACK and PACK only. up() and down() first wait until this thread's own engine work has left both
 *    the queue and the engine (CSR poll): for PACK that includes L1's write acknowledgments, so the other
 *    NEO can consume the data when the write mode acknowledges completion.
 *  - Only engine writes are ordered before up(). Data this thread writes to L1 with RISC stores is
 *    accepted by L1 before the post but may not be written yet when the other NEO reads it.
 *
 * Canonical ring use, capacity = ring depth:
 *   PACK   (producer NEO): wait_not_full(); pack_tile(ring, slot); up(1);
 *   UNPACK (consumer NEO): wait_min(1); copy_tile(ring, slot); down(1);
 *
 * Methods (same names as Semaphore):
 *  - up(n): wait for this thread's engine work, then add n. Call wait_not_full(n) BEFORE writing
 *    the shared buffer; up() does not enforce capacity. Updates larger than 7 use multiple accesses.
 *  - down(n): wait for this thread's engine work and for the count to reach n, then subtract n.
 *  - wait_min(n): wait until the count is at least n. Does not modify it.
 *  - wait_not_full(n): wait until the count is at most capacity - n. Does not modify it.
 *  - set(value): initialize the count while no other thread uses the semaphore.
 *  - value(): the current count.
 * All waits are RISC polls. They block the calling thread.
 */
class ClusterSemaphore {
public:
    ALWI ClusterSemaphore(std::uint32_t index, std::uint32_t capacity) : index_(index), capacity_(capacity) {
        ASSERT(index < TENSIX_GLOBAL_SEM_USER_COUNT);
        ASSERT(capacity >= 1 && capacity <= TENSIX_GLOBAL_SEM_VALUE_MASK);
    }

    // wait_engine_done() fences first, so this thread's earlier stores are accepted by L1 before the post; loads
    // can pass stores. Accepted is not written: see the RISC store rule above.
    ALWI void up(std::uint32_t n) {
        ASSERT(n <= capacity_);
        wait_engine_done();
        for (std::uint32_t step; n > 0; n -= step) {
            step = n < kMaxStep ? n : kMaxStep;
            tensix_global_sem_fetch_add(index_, step);
        }
    }

    ALWI void down(std::uint32_t n) {
        ASSERT(n <= capacity_);
        wait_engine_done();
        wait_min(n);
        for (std::uint32_t step; n > 0; n -= step) {
            step = n < kMaxStep ? n : kMaxStep;
            tensix_global_sem_fetch_sub(index_, step);
        }
    }

    // No fence after a poll: the loop's branch needs the read, so this in-order core issues nothing after it
    // (engine instructions included) until the read returns.
    ALWI void wait_min(std::uint32_t n) {
        ASSERT(n <= capacity_);
        WAYPOINT("CSMW");
        while (value() < n) {
        }
        WAYPOINT("CSMD");
    }

    ALWI void wait_not_full(std::uint32_t n = 1) {
        ASSERT(n <= capacity_);
        WAYPOINT("CSFW");
        while (value() > capacity_ - n) {
        }
        WAYPOINT("CSFD");
    }

    ALWI void set(std::uint32_t value) {
        ASSERT(value <= capacity_);
        tensix_global_sem_init(index_, value);
        fence();
    }

    ALWI std::uint32_t value() const { return tensix_global_sem_read(index_) & TENSIX_GLOBAL_SEM_VALUE_MASK; }

private:
    // The read aliases move the count by at most 7 per access.
    static constexpr std::uint32_t kMaxStep = 7;

    static ALWI void fence() { asm volatile("fence rw, rw" ::: "memory"); }

    // Wait until none of this thread's work for its replay buffer, MOP expander, or engine (packer or
    // unpacker) is queued or running.
    static ALWI void wait_engine_done() {
        bstatus_u mine{};
        mine.replay = 1;
        mine.mop = 1;
#if defined(TRISC_PACK)
        mine.pack = 1;
#else
        mine.unpack = 1;
#endif
        fence();
        while ((csr_read<CSR::tensix_busy_status, false>() & mine.val) != 0) {
        }
    }

    std::uint32_t index_;
    std::uint32_t capacity_;
};

}  // namespace ckernel
#endif  // ARCH_QUASAR && (TRISC_UNPACK || TRISC_PACK)

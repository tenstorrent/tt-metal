// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/metal_bounded_wait.hpp
//
// Budgeted waits on a Metal RISC's own NoC writes, so a destination that never
// acknowledges ends the wait instead of hanging the core. Metal JIT builds only.

#ifndef X280_KERNELS_SHARED_METAL_BOUNDED_WAIT_HPP
#define X280_KERNELS_SHARED_METAL_BOUNDED_WAIT_HPP

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#if defined(COMPILE_FOR_ERISC)
#include "internal/ethernet/tt_eth_api.h"
#endif

namespace metal_wait {

/** This core's wall clock in its own raw ticks: the Ethernet tile's on an ERISC, the RISC wall clock otherwise. */
inline uint64_t now() {
#if defined(COMPILE_FOR_ERISC)
    return eth_read_wall_clock();
#else
    return get_timestamp();
#endif
}

/* Polls of the write counters between two reads of now() in write_barrier.
 * An ordinary wait ends inside the first burst, which reads no clock: the
 * burst is unrolled to a bare load and branch per poll, as tight as
 * noc_async_write_barrier's loop, since a slower poll loop notices the
 * acknowledgement later. */
inline constexpr uint32_t kPollsPerClockRead = 1024u;

/**
 * noc_async_write_barrier(noc) with a budget: waits until every non-posted
 * write this core issued on `noc` is acknowledged, then invalidates the L1
 * cache, or gives up once `budget` ticks of now() have passed since its first
 * kPollsPerClockRead polls. `*waited`, when given, is set on success to
 * whether the first poll found writes outstanding.
 * @return false on expiry, with the writes still outstanding.
 */
inline bool write_barrier(uint32_t noc, uint64_t budget, bool* waited = nullptr) {
    const bool outstanding = !ncrisc_noc_nonposted_writes_flushed(noc);
    // One burst of polls; true as soon as one finds the writes acknowledged.
    const auto burst = [noc] {
#pragma GCC unroll 8
        for (uint32_t i = 0; i < kPollsPerClockRead; ++i) {
            if (ncrisc_noc_nonposted_writes_flushed(noc)) {
                return true;
            }
        }
        return false;
    };
    if (outstanding && !burst()) {
        const uint64_t t0 = now();
        while (!burst()) {
            if (now() - t0 >= budget) {
                return false;
            }
        }
    }
    if (waited != nullptr) {
        *waited = outstanding;
    }
    invalidate_l1_cache();
    return true;
}

/**
 * Waits until every write this core issued on `noc` has read its source, so
 * the source may be overwritten, or `budget` ticks of now() pass.
 * @return false on expiry.
 */
inline bool writes_sent(uint32_t noc, uint64_t budget) {
    const uint64_t t0 = now();
    while (!ncrisc_noc_nonposted_writes_sent(noc)) {
        if (now() - t0 >= budget) {
            return false;
        }
    }
    return true;
}

}  // namespace metal_wait

#endif  // X280_KERNELS_SHARED_METAL_BOUNDED_WAIT_HPP

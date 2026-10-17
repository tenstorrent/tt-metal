// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Counter-pair arithmetic for a single-producer, single-consumer ring: the
 * distance between a producer's and a consumer's counter, and whether the pair
 * is consistent with the ring. eth_credit's pointers wrap at twice the ring depth and
 * packet_injection's counters at 2^32; the mask names which.
 */
#ifndef X280_CREDIT_COUNTER_H
#define X280_CREDIT_COUNTER_H

#include <stdint.h>

/**
 * Slots from a consumer counter `rd` to a producer counter `wr` that wrap at
 * mask + 1: (wr - rd) & mask. `mask` is 2^k - 1 (2N - 1 for pointers over
 * 0..2N-1 on an N-deep ring, UINT32_MAX for free-running u32 counters). Pure
 * arithmetic: it does not check that the pair is consistent.
 */
static inline uint32_t credit_used(uint32_t wr, uint32_t rd, uint32_t mask) { return (wr - rd) & mask; }

/**
 * Whether a counter pair is consistent for a ring `depth` slots deep: both
 * counters are below mask + 1 and credit_used(wr, rd, mask) <= depth. Writes
 * that distance to `*used` either way. A pair more than depth but less than
 * mask + 1 apart is refused.
 *
 * `depth` is the ring's depth for a producer/consumer pair; passing a smaller
 * one checks that a counter has not moved past it (packet_injection refuses a
 * consumer that moved past the producer's occupancy).
 */
static inline int credit_pair_ok(uint32_t wr, uint32_t rd, uint32_t mask, uint32_t depth, uint32_t* used) {
    *used = credit_used(wr, rd, mask);
    return wr <= mask && rd <= mask && *used <= depth;
}

#endif /* X280_CREDIT_COUNTER_H */

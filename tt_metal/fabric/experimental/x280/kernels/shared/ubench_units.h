// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The units the benchmarks lay out NoC-written buffers in, and the NoC's
 * congruence rule. Suites name their own slots and strides from these.
 */
#ifndef X280_UBENCH_UNITS_H
#define X280_UBENCH_UNITS_H

#include <stdint.h>

/* One writer per this many bytes of LIM, taken to be one LIM ECC granule. The
 * granule is an owner assumption (item 9, MEASUREMENT.md
 * "Unconfirmed hardware assumptions"); a different granule would change this
 * value. */
#define UBENCH_LIM_GRANULE_BYTES 8u

/* The X280 cache line. The host writes LIM in whole lines (write_available_lim
 * admits nothing else), which the tree takes to prime a line's ECC before a
 * peer's partial write (MEASUREMENT.md, item 9). */
#define UBENCH_LIM_LINE_BYTES 64u

/* A length-carrying NoC write whose data comes from L1 lands only when its
 * source and destination are congruent modulo this (C16,
 * WormholeB0/NoC/Alignment.md "When NOC_AT_LEN_BE contains a length"); on
 * Wormhole, data from any other address needs C32. Blackhole's
 * NoC/MemoryMap.md links an Alignment.md that is absent, and Blackhole's
 * noc_parameters.h sets NOC_L1_{READ,WRITE}_ALIGNMENT_BYTES to 16. Whether an
 * X280 store through an L2CPU TLB window is data from another address is
 * MEASUREMENT.md item 19. */
#define UBENCH_NOC_UNIT_BYTES 16u

/** Whether an L1-sourced NoC write from `src` to `dst` meets the congruence rule. */
static inline int ubench_noc_congruent(uint64_t src, uint64_t dst) {
    return (src % UBENCH_NOC_UNIT_BYTES) == (dst % UBENCH_NOC_UNIT_BYTES);
}

#endif /* X280_UBENCH_UNITS_H */

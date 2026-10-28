// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_measurement.h -- fenced measurement bracket for X280 microbenchmarks.
 *
 * X280 microbenchmarks bracket each timed sample
 * with a rdcycle read at start and end. rt/x280_hw.h supplies the raw counter
 * read; this header adds the fences the measurement contract requires.
 *
 * Two hazards these brackets avoid:
 *
 *   - The compiler is free to hoist code out of, or sink code into, a
 *     measurement window if nothing anchors the boundary. `fence io, io`
 *     names the I/O device space specifically so a preceding store to LIM or
 *     to a NOC window cannot slide past the rdcycle. It also has stronger
 *     ordering than the memory model asks for -- fine, because the cost of
 *     one fence per sample is small compared to the noise floor, and the
 *     alternative (a partial fence that happens to work today) is fragile.
 *
 *   - `rdcycle` is a counter at the core's operating frequency, not a wall
 *     clock. Two consequences for the measurement contract: (1) a `rdcycle`
 *     delta is a raw native CPU-cycle count, not elapsed seconds, and
 *     requires no frequency calibration for that output; (2)
 *     `rdcycle` is not portable across different-PLL cores in the same run,
 *     so a same-bracket delta on one hart is what this header measures --
 *     no cross-hart timestamps.
 *
 * The two-brackets pattern is:
 *
 *   uint64_t t0 = x280_rdcycle_fenced();
 *   ... code under measurement ...
 *   uint64_t t1 = x280_rdcycle_fenced();
 *   delta = t1 - t0;
 *
 * Two identical fenced reads at the boundary keep the header trivially
 * verifiable: no matter what the caller compiles to on either side, the
 * fences flank the counter read.
 */
#ifndef X280_RT_MEASUREMENT_H
#define X280_RT_MEASUREMENT_H

#include <stdint.h>

#include "rt/x280_hw.h"

/* Fenced counter read: a compiler barrier before, `rdcycle`, and a barrier
 * after. The count is executed core cycles, which do not advance while the
 * hart stalls (x280_rdcycle() in rt/x280_hw.h); elapsed time and timeouts use
 * x280_mtime_fenced(). Emitted as `fence io, io` on both sides so the fence names I/O
 * device space explicitly and cannot be optimised away as a plain memory
 * fence would sometimes let happen inside a hot loop.
 *
 * `__attribute__((always_inline))` is essential: a call boundary here would
 * defeat the point of an inline counter read, and both callers -- the timed
 * loop's start and end -- are hot. Marked `static` because inlining across
 * translation units is not required (every timed loop lives in one image).
 */
static inline __attribute__((always_inline)) uint64_t x280_rdcycle_fenced(void) {
    uint64_t c;
    x280_fence_io();
    __asm__ volatile("rdcycle %0" : "=r"(c));
    x280_fence_io();
    return c;
}

/* Fenced wall-clock read: for cases where a PLL-independent reference is
 * preferred to a `rdcycle` reading at the running PLL -- for example a
 * bandwidth bracket where the batch time is dominated by a
 * NOC read whose latency is set by the fabric, not by
 * the core's operating frequency, and reporting in seconds against a fixed
 * 50 MHz reference is more useful than reporting in cycles that then need
 * a PLL conversion on the host. `x280_mtime` reads the
 * CLINT wall clock (fixed 50 MHz, PLL-independent, see the note beside it
 * in rt/x280_hw.h).
 *
 * The fence pattern is the same as `x280_rdcycle_fenced`: `fence io, io`
 * on both sides names the I/O device space explicitly (the mtime CSR is
 * a device-space memory-mapped load) and cannot be optimised away as a
 * plain memory fence would sometimes let happen inside a hot loop.
 *
 * 20 ns tick length is coarse compared to `rdcycle`, but a batch of
 * >= 10 µs spans hundreds of mtime ticks, so the resolution is not the
 * bottleneck there.
 */
static inline __attribute__((always_inline)) uint64_t x280_mtime_fenced(void) {
    uint64_t t;
    x280_fence_io();
    t = *(volatile uint64_t*)X280_MTIME_ADDR;
    x280_fence_io();
    return t;
}

/* Both counters at one end of a paired bracket. `wall` is the elapsed
 * reference; `cpu` counts executed core cycles, which do not advance while the
 * hart stalls, so a `cpu` delta is executed cost, not elapsed time. */
typedef struct {
    uint64_t wall; /* x280_mtime_fenced() */
    uint64_t cpu;  /* x280_rdcycle_fenced() */
} x280_paired_t;

/* Reads a paired bracket end, wall then CPU: the one order of every X280
 * paired bracket. */
static inline __attribute__((always_inline)) x280_paired_t x280_paired_now(void) {
    x280_paired_t t;
    t.wall = x280_mtime_fenced();
    t.cpu = x280_rdcycle_fenced();
    return t;
}

#endif /* X280_RT_MEASUREMENT_H */

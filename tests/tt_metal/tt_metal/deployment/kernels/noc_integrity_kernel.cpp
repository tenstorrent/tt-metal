// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "args.hpp"
#include "noc_integrity_types.hpp"
#include "patterns/patterns.hpp"

#define ARGS(X)                \
    X(uint32_t, progress_addr) \
    X(uint32_t, results_addr)  \
    X(uint32_t, src_addr)      \
    X(uint32_t, readback_addr) \
    X(uint32_t, slice_size)    \
    X(uint32_t, iterations)    \
    X(uint32_t, timeout_ms)    \
    X(uint32_t, packet_size)   \
    X(uint32_t, burst)

// Assumes a ~1.35 GHz wall clock (same as eth/common.hpp)
constexpr uint64_t CYCLES_PER_MS = 1350000ull;

#define WALL_CLOCK_L (*(volatile uint32_t*)0xFFB121F0)
#define WALL_CLOCK_H_LIVE (*(volatile uint32_t*)0xFFB121F4)

// The wall clock, read high-low-high from the live high word and retried if the high word changed. The latched
// high word that timestamp.hpp reads (0xFFB121F8) keeps tracking the clock until the tile's next register
// access, so low-then-high tears by +2^32 cycles when the low word wraps between the two reads. That ended a
// barrier wait early as a spurious 2 s timeout.
static uint64_t wall_clock() {
    for (;;) {
        const uint32_t high = WALL_CLOCK_H_LIVE;
        const uint32_t low = WALL_CLOCK_L;
        if (WALL_CLOCK_H_LIVE == high) {
            return ((uint64_t)high << 32) | low;
        }
    }
}

static bool wait_writes_flushed(uint64_t timeout_cycles) {
    const uint64_t start = wall_clock();
    for (;;) {
        invalidate_l1_cache();
        bool done;
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            done = ncrisc_dynamic_noc_nonposted_writes_flushed(noc_index);
        } else {
            done = ncrisc_noc_nonposted_writes_flushed(noc_index);
        }
        if (done) {
            return true;
        }
        if (wall_clock() - start > timeout_cycles) {
            return false;
        }
    }
}

static bool wait_reads_flushed(uint64_t timeout_cycles) {
    const uint64_t start = wall_clock();
    for (;;) {
        invalidate_l1_cache();
        bool done;
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            done = ncrisc_dynamic_noc_reads_flushed(noc_index);
        } else {
            done = ncrisc_noc_reads_flushed(noc_index);
        }
        if (done) {
            invalidate_l1_cache();
            return true;
        }
        if (wall_clock() - start > timeout_cycles) {
            return false;
        }
    }
}

static void fill_pattern(
    volatile uint32_t* buf, uint32_t words, uint32_t pattern, uint32_t iter, uint32_t dst_addr, uint32_t prbs_seed) {
    switch (pattern) {
        case NOC_PATTERN_WALKING_ONES:
            for (uint32_t i = 0; i < words; i++) {
                buf[i] = dram_pattern_marching_ones(iter, i);
            }
            break;
        case NOC_PATTERN_WALKING_ZEROES:
            for (uint32_t i = 0; i < words; i++) {
                buf[i] = dram_pattern_marching_zeroes(iter, i);
            }
            break;
        case NOC_PATTERN_CHECKERBOARD:
        case NOC_PATTERN_CHECKERBOARD_INV:
            for (uint32_t i = 0; i < words; i++) {
                buf[i] = dram_pattern_checkerboard(pattern == NOC_PATTERN_CHECKERBOARD_INV, i);
            }
            break;
        case NOC_PATTERN_ADDRESS:
            for (uint32_t i = 0; i < words; i++) {
                buf[i] = dram_pattern_address(iter, (dst_addr >> 2) + i);
            }
            break;
        case NOC_PATTERN_PRBS:
        default: {
            uint32_t state = prbs_seed ? prbs_seed : 1u;
            for (uint32_t i = 0; i < words; i++) {
                state = dram_pattern_random_step(state);
                buf[i] = state;
            }
            break;
        }
    }
}

// A slice is one source's own region in a destination's L1 (or DRAM buffer), slice_size bytes at the source's
// slot, so concurrent sources never write the same bytes. These move a whole slice as packets of packet_bytes.
//
// Unlinked writes go through noc_async_write, so the watcher and NOC tracing see them. The dataflow API has no
// linked unicast write, so a linked slice issues every packet through the low-level call, with the VC linked on
// all but the last; the chain holds its VC from the first packet to the last. The last packet can't go through
// noc_async_write either: with the watcher's linked-transaction check on, it hangs any unicast issued while the
// previous packet left the link set.
static void write_slice(
    uint32_t src_addr, uint64_t dst, uint32_t slice_bytes, uint32_t packet_bytes, uint32_t vc, bool linked) {
    for (uint32_t offset = 0; offset < slice_bytes; offset += packet_bytes) {
        const uint32_t len = slice_bytes - offset < packet_bytes ? slice_bytes - offset : packet_bytes;
        if (linked) {
            ncrisc_noc_fast_write_any_len<noc_mode>(
                noc_index,
                write_cmd_buf,
                src_addr + offset,
                dst + offset,
                len,
                vc,
                false,
                offset + len < slice_bytes,
                1,
                false);
        } else {
            noc_async_write(src_addr + offset, dst + offset, len, noc_index, vc);
        }
    }
}

// The VC noc_init leaves on the read command buffer.
constexpr uint32_t NOC_DEFAULT_READ_VC = 1;

// In dedicated NOC mode noc_async_read ignores its VC argument and issues on whatever VC the read command buffer
// holds, so set it first, and put the default back after so later kernels on this core read as they expect.
static void read_slice(uint64_t src, uint32_t dst_addr, uint32_t slice_bytes, uint32_t packet_bytes, uint32_t vc) {
    noc_async_read_one_packet_set_state<true>(src, packet_bytes, vc, noc_index);
    for (uint32_t offset = 0; offset < slice_bytes; offset += packet_bytes) {
        const uint32_t len = slice_bytes - offset < packet_bytes ? slice_bytes - offset : packet_bytes;
        noc_async_read(src + offset, dst_addr + offset, len, noc_index, vc);
    }
    noc_async_read_one_packet_set_state<true>(src, packet_bytes, NOC_DEFAULT_READ_VC, noc_index);
}

// Records where a barrier timed out and marks the kernel aborted; the caller then returns.
static void abort_on_timeout(
    volatile uint32_t* progress, volatile uint32_t* r, uint32_t status, uint32_t pattern, uint32_t iter, uint32_t txn) {
    r[NOC_RESULT_STATUS] = status;
    r[NOC_RESULT_PATTERN] = pattern;
    r[NOC_RESULT_ITERATION] = iter;
    r[NOC_RESULT_TXN] = txn;
    progress[NOC_PROGRESS_STATE] = NOC_STATE_ABORTED;
}

void kernel_main() {
    ARG_INIT(ARGS);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t num_dests = get_arg_val<uint32_t>(1);
    const uint32_t words = slice_size / sizeof(uint32_t);
    const uint64_t timeout_cycles = (uint64_t)timeout_ms * CYCLES_PER_MS;

    volatile uint32_t* progress = (volatile uint32_t*)progress_addr;
    volatile uint32_t* results = (volatile uint32_t*)results_addr;
    volatile uint32_t* src = (volatile uint32_t*)src_addr;
    volatile uint32_t* rb = (volatile uint32_t*)readback_addr;

    progress[NOC_PROGRESS_HEARTBEAT] = 0;
    progress[NOC_PROGRESS_STATE] = NOC_STATE_RUNNING;

    for (uint32_t d = 0; d < num_dests; d++) {
        volatile uint32_t* r = results + d * NOC_RESULT_WORDS;
        for (uint32_t w = 0; w < NOC_RESULT_WORDS; w++) {
            r[w] = 0;
        }
        r[NOC_RESULT_STATUS] = NOC_STATUS_NOT_RUN;
    }

    // Destination-outer, so a later abort can never leave a partially tested destination marked done.
    for (uint32_t d = 0; d < num_dests; d++) {
        const uint32_t x = get_arg_val<uint32_t>(2 + d * 3);
        const uint32_t y = get_arg_val<uint32_t>(3 + d * 3);
        const uint32_t dst_addr = get_arg_val<uint32_t>(4 + d * 3);
        const uint64_t dst_noc_addr = get_noc_addr(x, y, dst_addr);
        volatile uint32_t* r = results + d * NOC_RESULT_WORDS;
        progress[NOC_PROGRESS_DEST] = d;

        for (uint32_t iter = 0; iter < iterations; iter++) {
            progress[NOC_PROGRESS_ITERATION] = iter;
            // Transaction-outer, so consecutive transfers to a destination always carry different data, and a
            // write that never lands can't pass by reading back the previous transfer's identical bytes.
            for (uint32_t txn = 0; txn < NOC_TXNS; txn++) {
                const uint32_t vc = txn & NOC_TXN_VC_MASK;
                const bool linked = txn & NOC_TXN_LINKED;
                const uint32_t packet_bytes = (txn & NOC_TXN_SMALL) ? NOC_SMALL_PACKET : packet_size;
                progress[NOC_PROGRESS_TXN] = txn;

                for (uint32_t pattern = 0; pattern < NOC_PATTERN_COUNT; pattern++) {
                    progress[NOC_PROGRESS_PATTERN] = pattern;

                    fill_pattern(
                        src, words, pattern, iter, dst_addr, seed ^ (d << 16) ^ (iter << 8) ^ (txn << 3) ^ pattern);
                    for (uint32_t i = 0; i < words; i++) {
                        rb[i] = ~src[i];
                    }

                    // NOC_TEST_BURST > 1 rewrites the slice that many times before the barrier, keeping more packets in
                    // flight, like the fabric EDM does when forwarding to local cores.
                    for (uint32_t b = 0; b < burst; b++) {
                        write_slice(src_addr, dst_noc_addr, slice_size, packet_bytes, vc, linked);
                    }
                    if (!wait_writes_flushed(timeout_cycles)) {
                        abort_on_timeout(progress, r, NOC_STATUS_WRITE_TIMEOUT, pattern, iter, txn);
                        return;
                    }

                    // Read back on the same VC and packet size.
                    read_slice(dst_noc_addr, readback_addr, slice_size, packet_bytes, vc);
                    if (!wait_reads_flushed(timeout_cycles)) {
                        abort_on_timeout(progress, r, NOC_STATUS_READ_TIMEOUT, pattern, iter, txn);
                        return;
                    }

                    uint32_t mismatches = 0;
                    for (uint32_t i = 0; i < words; i++) {
                        const uint32_t expected = src[i];
                        const uint32_t actual = rb[i];
                        if (expected != actual) {
                            if (r[NOC_RESULT_MISMATCHES] == 0 && mismatches == 0) {
                                r[NOC_RESULT_FIRST_OFFSET] = i * sizeof(uint32_t);
                                r[NOC_RESULT_EXPECTED] = expected;
                                r[NOC_RESULT_ACTUAL] = actual;
                                r[NOC_RESULT_PATTERN] = pattern;
                                r[NOC_RESULT_ITERATION] = iter;
                                r[NOC_RESULT_TXN] = txn;
                            }
                            mismatches++;
                        }
                    }
                    r[NOC_RESULT_MISMATCHES] = r[NOC_RESULT_MISMATCHES] + mismatches;
                    if (mismatches) {
                        r[NOC_RESULT_TXN_MASK] = r[NOC_RESULT_TXN_MASK] | (1u << txn);
                    }
                    r[NOC_RESULT_TRANSFERS] = r[NOC_RESULT_TRANSFERS] + 1;
                    progress[NOC_PROGRESS_HEARTBEAT] = progress[NOC_PROGRESS_HEARTBEAT] + 1;
                }
            }
        }
        r[NOC_RESULT_STATUS] = NOC_STATUS_DONE;
    }

    progress[NOC_PROGRESS_STATE] = NOC_STATE_DONE;
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "args.hpp"
#include "timestamp.hpp"
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
    X(uint32_t, burst)         \
    X(uint32_t, vc)

// Assumes a ~1.35 GHz wall clock (same as eth/common.hpp)
constexpr uint64_t CYCLES_PER_MS = 1350000ull;

static bool wait_writes_flushed(uint64_t timeout_cycles) {
    const uint64_t start = timestamp();
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
        if (timestamp() - start > timeout_cycles) {
            return false;
        }
    }
}

static bool wait_reads_flushed(uint64_t timeout_cycles) {
    const uint64_t start = timestamp();
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
        if (timestamp() - start > timeout_cycles) {
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
        r[NOC_RESULT_STATUS] = NOC_STATUS_NOT_RUN;
        r[NOC_RESULT_MISMATCHES] = 0;
        r[NOC_RESULT_FIRST_OFFSET] = 0;
        r[NOC_RESULT_EXPECTED] = 0;
        r[NOC_RESULT_ACTUAL] = 0;
        r[NOC_RESULT_PATTERN] = 0;
        r[NOC_RESULT_ITERATION] = 0;
        r[NOC_RESULT_TRANSFERS] = 0;
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
            for (uint32_t pattern = 0; pattern < NOC_PATTERN_COUNT; pattern++) {
                progress[NOC_PROGRESS_PATTERN] = pattern;
                progress[NOC_PROGRESS_ITERATION] = iter;

                fill_pattern(src, words, pattern, iter, dst_addr, seed ^ (d << 16) ^ (iter << 8) ^ pattern);
                for (uint32_t i = 0; i < words; i++) {
                    rb[i] = ~src[i];
                }

                // Keep many packets in flight, like the fabric EDM does when forwarding to local cores.
                for (uint32_t b = 0; b < burst; b++) {
                    for (uint32_t off = 0; off < slice_size; off += packet_size) {
                        const uint32_t len = slice_size - off < packet_size ? slice_size - off : packet_size;
                        noc_async_write(src_addr + off, dst_noc_addr + off, len, noc_index, vc);
                    }
                }
                if (!wait_writes_flushed(timeout_cycles)) {
                    r[NOC_RESULT_STATUS] = NOC_STATUS_WRITE_TIMEOUT;
                    r[NOC_RESULT_PATTERN] = pattern;
                    r[NOC_RESULT_ITERATION] = iter;
                    progress[NOC_PROGRESS_STATE] = NOC_STATE_ABORTED;
                    return;
                }

                noc_async_read(dst_noc_addr, readback_addr, slice_size);
                if (!wait_reads_flushed(timeout_cycles)) {
                    r[NOC_RESULT_STATUS] = NOC_STATUS_READ_TIMEOUT;
                    r[NOC_RESULT_PATTERN] = pattern;
                    r[NOC_RESULT_ITERATION] = iter;
                    progress[NOC_PROGRESS_STATE] = NOC_STATE_ABORTED;
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
                        }
                        mismatches++;
                    }
                }
                r[NOC_RESULT_MISMATCHES] = r[NOC_RESULT_MISMATCHES] + mismatches;
                r[NOC_RESULT_TRANSFERS] = r[NOC_RESULT_TRANSFERS] + 1;
                progress[NOC_PROGRESS_HEARTBEAT] = progress[NOC_PROGRESS_HEARTBEAT] + 1;
            }
        }
        r[NOC_RESULT_STATUS] = NOC_STATUS_DONE;
    }

    progress[NOC_PROGRESS_STATE] = NOC_STATE_DONE;
}

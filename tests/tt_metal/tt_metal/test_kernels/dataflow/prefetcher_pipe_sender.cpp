// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe sender kernel: push `num_entries` dense entries from a host-populated L1 staging
// area using one of the sender write primitives.
//
// Bindings:
//   pipe::out              — KernelSpec::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::entry_size       - bytes per entry (must be L1_ALIGNMENT multiple)
//   args::num_entries      - number of entries to push per receiver
//   args::write_primitive  - 0=write_broadcast, 1=write_strided,
//                            2=write_to_receiver(r)+push_back (1:1 uses r=0),
//                            3=write_to_receiver+push_back_to_receiver (per-receiver credit),
//                            4=decoupled: reserve(n) + write_broadcast(n) + flush + push_back(n),
//                            5=per-receiver credit interleaved across receivers (entry-major)
//   args::data_pattern     - 0=multicast counter layout, 1=strided per-receiver layout,
//                            2=per-receiver constant layout (see prefetcher_pipe_test_utils.hpp)
//   args::do_barrier       - 1 to call barrier() after pushing all entries
// Args (named RTAs):
//   args::staging_addr     - sender-local L1 scratch region pre-populated by the host
//
// Quasar multi-DM: launched with num_threads > 1 every hart runs this body; the pipe APIs
// partition receivers by hart (Flow C), so the same loop is correct for any thread count.

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t write_primitive = get_arg(args::write_primitive);
    constexpr uint32_t data_pattern = get_arg(args::data_pattern);
    constexpr uint32_t do_barrier = get_arg(args::do_barrier);

    // Must match SenderDataPattern in prefetcher_pipe_test_utils.hpp.
    constexpr uint32_t pattern_multicast_counter = 0;
    constexpr uint32_t pattern_strided_per_receiver = 1;
    constexpr uint32_t pattern_per_receiver_constant = 2;

    const uint32_t staging_base = get_arg(args::staging_addr);
    const CoreLocalMem<uint8_t> staging(staging_base);

    Noc noc;
    experimental::PrefetcherPipe gdfb(pipe::out);

    static_assert(
        write_primitive != 0 || data_pattern == pattern_multicast_counter,
        "write_broadcast expects multicast counter staging");
    static_assert(
        write_primitive != 1 || data_pattern == pattern_strided_per_receiver, "write_strided expects strided staging");
    static_assert(
        write_primitive != 2 || data_pattern == pattern_per_receiver_constant,
        "write_to_receiver expects per-receiver staging");
    static_assert(
        write_primitive != 3 || data_pattern == pattern_per_receiver_constant,
        "push_back_to_receiver expects per-receiver staging");
    static_assert(
        write_primitive != 4 || data_pattern == pattern_multicast_counter,
        "decoupled write_broadcast expects multicast counter staging");
    static_assert(
        write_primitive != 5 || data_pattern == pattern_multicast_counter,
        "interleaved per-receiver credit expects multicast counter staging");

    if constexpr (write_primitive == 0) {
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            gdfb.write_broadcast(noc, staging, 1, {.offset_bytes = i * entry_size});
            gdfb.flush_writes(noc);
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 1) {
        const uint32_t num_recv = gdfb.num_receivers();
        const uint32_t row_bytes = num_recv * entry_size;
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            gdfb.write_strided(noc, staging, 1, 1, entry_size, {.offset_bytes = i * row_bytes});
            gdfb.flush_writes(noc);
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 2) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            for (uint32_t r = 0; r < num_recv; ++r) {
                gdfb.write_to_receiver(noc, r, staging, 1, {.offset_bytes = r * entry_size});
            }
            gdfb.flush_writes(noc);
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 3) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t r = 0; r < num_recv; ++r) {
            for (uint32_t i = 0; i < num_entries; ++i) {
                gdfb.reserve_back_for_receiver(r, 1);
                gdfb.write_to_receiver(noc, r, staging, 1, {.offset_bytes = r * entry_size});
                gdfb.flush_writes(noc);
                gdfb.push_back_to_receiver(r, 1, noc);
            }
        }
    } else if constexpr (write_primitive == 4) {
        // Layered contract: all payload writes land before any pages_sent credit.
        // write_* does not advance fifo_wr_ptr, so one write_broadcast(n) covers the slot;
        // a single push_back(n) then publishes credit for the whole batch.
        gdfb.reserve_back(num_entries);
        gdfb.write_broadcast(noc, staging, num_entries);
        gdfb.flush_writes(noc);
        gdfb.push_back(num_entries, noc);
    } else if constexpr (write_primitive == 5) {
        // Entry-major per-receiver credit: every receiver gets entry i before entry i+1.
        // Each receiver must still land entry i in its own slot i, which only holds if the
        // sender derives an independent write position per receiver from its credits
        // rather than sharing one cursor for the slot.
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t i = 0; i < num_entries; ++i) {
            for (uint32_t r = 0; r < num_recv; ++r) {
                gdfb.reserve_back_for_receiver(r, 1);
                gdfb.write_to_receiver(noc, r, staging, 1, {.offset_bytes = i * entry_size});
                gdfb.flush_writes(noc);
                gdfb.push_back_to_receiver(r, 1, noc);
            }
        }
    }

    if constexpr (do_barrier) {
        gdfb.barrier();
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PersistentDFB sender kernel
//
// Compile-time parameters (via kernel compile_args):
//   [0] persistent_dfb_id      - runtime-assigned slot (CreatePersistentDFB persistent_dfb_id on host)
//   [1] entry_size         - bytes per entry (must be L1_ALIGNMENT multiple)
//   [2] num_entries        - number of entries to push per receiver
//   [3] write_primitive    - 0=write_broadcast, 1=write_strided,
//                            2=write_to_receiver(r)+push_back (1:1 uses r=0),
//                            3=write_to_receiver+push_back_to_receiver (per-receiver credit),
//                            4=decoupled: reserve(n) + write_broadcast(n) + flush + push_back(n),
//                            5=per-receiver credit interleaved across receivers (entry-major)
//   [4] data_pattern       - 0=multicast counter layout, 1=strided per-receiver layout,
//                            2=per-receiver constant layout (see persistent_dfb_test_utils.hpp)
//   [5] do_barrier         - 1 to call barrier() after pushing all entries
//
// Runtime args:
//   [0] l1_staging_addr    - sender-local L1 scratch region pre-populated by the host

#include "api/dataflow/persistent_dfb.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

FORCE_INLINE uint32_t staging_addr(uint32_t staging_base, uint32_t byte_offset) { return staging_base + byte_offset; }

void kernel_main() {
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);
    constexpr uint32_t write_primitive = get_compile_time_arg_val(3);
    constexpr uint32_t data_pattern = get_compile_time_arg_val(4);
    constexpr uint32_t do_barrier = get_compile_time_arg_val(5);

    // Must match SenderDataPattern in persistent_dfb_test_utils.hpp.
    constexpr uint32_t pattern_multicast_counter = 0;
    constexpr uint32_t pattern_strided_per_receiver = 1;
    constexpr uint32_t pattern_per_receiver_constant = 2;

    const uint32_t staging_base = get_arg_val<uint32_t>(0);

    Noc noc;
    // Spot-check: log first byte of each entry (host pre-populated staging)
    DPRINT("l1_staging_addr: 0x{:x}\n", staging_base);

    experimental::PersistentDFB gdfb(persistent_dfb_id);

    DPRINT("Running write_primitive: {}\n", write_primitive);

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
            DPRINT("Reserving back for broadcast\n");
            gdfb.reserve_back(1);
            DPRINT("Done reserve back for broadcast to {}\n", staging_addr(staging_base, i * entry_size));
            gdfb.write_broadcast(staging_addr(staging_base, i * entry_size), 1, noc);
            DPRINT("Done write broadcast\n");
            gdfb.flush_writes(noc);
            DPRINT("Done posted write flush\n");
            gdfb.push_back(1, noc);
            DPRINT("Done push back\n");
        }
    } else if constexpr (write_primitive == 1) {
        const uint32_t num_recv = gdfb.num_receivers();
        const uint32_t row_bytes = num_recv * entry_size;
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            gdfb.write_strided(staging_addr(staging_base, i * row_bytes), 1, 1, entry_size, noc);
            gdfb.flush_writes(noc);
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 2) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            for (uint32_t r = 0; r < num_recv; ++r) {
                gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
            }
            gdfb.flush_writes(noc);
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 3) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t r = 0; r < num_recv; ++r) {
            for (uint32_t i = 0; i < num_entries; ++i) {
                gdfb.reserve_back_for_receiver(r, 1);
                gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
                gdfb.flush_writes(noc);
                gdfb.push_back_to_receiver(r, 1, noc);
            }
        }
    } else if constexpr (write_primitive == 4) {
        // Layered contract: all payload writes land before any pages_sent credit.
        // write_* does not advance fifo_wr_ptr, so one write_broadcast(n) covers the slot;
        // a single push_back(n) then publishes credit for the whole batch.
        gdfb.reserve_back(num_entries);
        gdfb.write_broadcast(staging_base, num_entries, noc);
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
                gdfb.write_to_receiver(r, staging_addr(staging_base, i * entry_size), 1, noc);
                gdfb.flush_writes(noc);
                gdfb.push_back_to_receiver(r, 1, noc);
            }
        }
    }

    if constexpr (do_barrier) {
        gdfb.barrier();
    }
}

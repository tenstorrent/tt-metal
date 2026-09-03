// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke-test DRISC sender for a DRAM-sender PrefetcherPipe.
//
// Pushes a host-preloaded pattern from DRISC L1 to each receiver, one entry at a time, using the
// bare sender helpers in internal/prefetcher_pipe_dram_sender.h. Unlike the GlobalCircularBuffer
// smoke sender, nothing here stands up a mock config block: the host has already written a real
// PrefetcherPipe sender config page into DRISC L1, so the kernel just loads it. Receivers are
// ordinary workers running the device PrefetcherPipe class.
//
// Compile-time args:
//   [0] config_page_addr  - DRISC L1 address of this sender's PrefetcherPipe config page
//   [1] num_entries       - entries to push per receiver
//   [2] data_l1_base      - DRISC L1 base of the host-preloaded pattern
//
// The pattern is laid out per receiver: receiver r's entry i is at
// data_l1_base + (r * num_entries + i) * entry_size, so each receiver gets distinct bytes and a
// mis-addressed write shows up as the wrong receiver's data rather than as silence.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/drisc_mode.h"
#include "internal/prefetcher_pipe_dram_sender.h"

// DRISC firmware does not define cb_interface (no CB infrastructure on DRAM cores), and
// dataflow_api.h references it.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t config_page_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(1);
    constexpr uint32_t data_l1_base = get_compile_time_arg_val(2);

    experimental::PipeSenderCtx ctx;
    experimental::pipe_load_sender_ctx(ctx, config_page_addr);

    // DRISC needs stream mode for NIU-initiated NoC traffic, including the credit atomics the
    // receivers send back to this core's counters.
    experimental::drisc_set_stream_mode();

    for (uint32_t i = 0; i < num_entries; ++i) {
        experimental::pipe_reserve_back(ctx, 1);
        for (uint32_t r = 0; r < ctx.num_receivers; ++r) {
            const uint32_t src = data_l1_base + (r * num_entries + i) * ctx.entry_bytes;
            experimental::pipe_write_to_receiver(ctx, r, src, 1, noc_index);
        }
        // Payload must land before the credit that advertises it.
        noc_async_posted_writes_flushed();
        experimental::pipe_push_credits(ctx, 1, noc_index);
    }

    // Drain: every receiver has consumed and acked everything. Acks target DRISC L1, which is only
    // reachable while this core is in stream mode, so the barrier has to precede the mode restore.
    experimental::pipe_sender_barrier(ctx);
    experimental::drisc_set_noc2axi_mode();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke-test DRISC sender for a DRAM-sender PrefetcherPipe.
//
// Pushes a host-preloaded pattern from DRISC L1 to each receiver, one entry at a time, through the
// device PrefetcherPipe class -- the same class the worker receivers consume with. A DRAM core has
// no Program slot to name its pipe, so the kernel builds it from the sender config page the host
// stamped into DRISC L1.
//
// Compile-time args:
//   [0] config_page_addr  - DRISC L1 address of this sender's PrefetcherPipe config page
//   [1] num_entries       - entries to push per receiver
//   [2] data_l1_base      - DRISC L1 base of the host-preloaded pattern
//   [3] entry_bytes       - push granularity for this batch; may differ from the size the pipe was
//                           created with, which is what makes this a block-size change
//
// The pattern is laid out per receiver: receiver r's entry i is at
// data_l1_base + (r * num_entries + i) * entry_size, so each receiver gets distinct bytes and a
// mis-addressed write shows up as the wrong receiver's data rather than as silence.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"

// DRISC firmware does not define cb_interface (no CB infrastructure on DRAM cores), and
// dataflow_api.h references it.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t config_page_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(1);
    constexpr uint32_t data_l1_base = get_compile_time_arg_val(2);
    constexpr uint32_t entry_bytes = get_compile_time_arg_val(3);

    // Snaps onto this batch's entry grid and publishes the pad credits the receivers' own resize
    // waits for, when entry_bytes differs from the size last applied.
    experimental::PrefetcherPipe pipe(experimental::DramSenderConfigPage{config_page_addr}, entry_bytes);
    const CoreLocalMem<uint8_t> pattern(data_l1_base);
    Noc noc;

    const uint32_t num_receivers = pipe.num_receivers();
    for (uint32_t i = 0; i < num_entries; ++i) {
        pipe.reserve_back(1);
        for (uint32_t r = 0; r < num_receivers; ++r) {
            pipe.write_to_receiver(noc, r, pattern, 1, {.offset_bytes = (r * num_entries + i) * entry_bytes});
        }
        // Payload must land before the credit that advertises it.
        pipe.flush_writes(noc);
        pipe.push_back(1, noc);
    }

    // Drain: every receiver has consumed and acked everything before this kernel returns. Their
    // acks target this core's L1, and firmware leaves the NIU in stream mode for the board's
    // lifetime, so there is no mode restore for the barrier to be sequenced against.
    pipe.barrier();
}

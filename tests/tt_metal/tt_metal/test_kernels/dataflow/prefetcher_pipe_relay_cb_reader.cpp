// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM driver for a PrefetcherPipe relayed through a classic circular buffer.
//
// The CB was created over the pipe ring with experimental::CreateCircularBuffer(..., pipe, id), so
// its pages are the delivered bytes themselves and this kernel moves none of them: it only turns a
// delivered entry into CB credit for compute, and a compute pop back into a pipe ack.
//
// This is the loop shape the 1D matmul's in1 reader runs: publish the current entry, then let
// pop_front wait for compute to finish the previous one, so one entry of lookahead stays in flight.
// The CB is published exactly once per entry -- publishing again through the RelayView would hand
// compute twice the credit for the same bytes.
//
// Compile-time args:
//   [0] prefetcher_pipe_id
//   [1] cb_id             - relay circular buffer index, paged at the delivered entry size
//   [2] num_entries       - entries to consume

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"

void kernel_main() {
    constexpr uint8_t prefetcher_pipe_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);

    Noc noc;
    experimental::PrefetcherPipe pipe(prefetcher_pipe_id);
    // Aligns the CB to the pipe's durable cursor and arms pop_front's wait on compute. The returned
    // producer view is deliberately unused: publishing happens through the CB below.
    pipe.bind_relay();
    DataflowBuffer relay_cb(cb_id);

    for (uint32_t entry = 0; entry < num_entries; ++entry) {
        relay_cb.reserve_back(1);
        pipe.wait_front(entry == 0 ? 1u : 2u);
        relay_cb.push_back(1);
        if (entry >= 1) {
            pipe.pop_front(1, noc);
        }
    }
    // The loop leaves the last entry unacked so it stays published while compute drains it.
    if constexpr (num_entries > 0) {
        pipe.pop_front(1, noc);
    }
}

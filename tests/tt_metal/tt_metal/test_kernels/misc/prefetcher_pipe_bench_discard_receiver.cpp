// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bandwidth-bench receiver for PrefetcherPipe delivery: the PrefetcherPipe counterpart of
// gcb_bench_discard_receiver.cpp. Per-entry wait_front + pop_front in a loop, discards the data.
//
// Draining one entry at a time (rather than one wait_front(num_iters)) is what keeps the ring --
// which holds only a few entries of in-flight data -- refilling as the sender pushes through
// num_iters entries, so the measured rate is the sender's and not the ring's capacity.
//
// No barrier at exit: the acks pop_front posts are what the sender's stop barrier waits on, and the
// durable read cursor is checkpointed by PrefetcherPipe::commit() when the object goes out of scope.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_compile_time_arg_val(0);

    // One kernel serves the receivers of every pipe, and a core's pipe id depends on which sender
    // drives it, so the id is a runtime arg rather than a compile-time one.
    const uint8_t prefetcher_pipe_id = static_cast<uint8_t>(get_arg_val<uint32_t>(0));

    Noc noc;
    experimental::PrefetcherPipe pipe(prefetcher_pipe_id);

    for (uint32_t i = 0; i < num_iters; ++i) {
        pipe.wait_front(1);
        pipe.pop_front(1, noc);
    }
}

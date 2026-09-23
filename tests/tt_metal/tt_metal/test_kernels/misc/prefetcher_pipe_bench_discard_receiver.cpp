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
// A nonzero hold_cycles delays the first pop, so a test can hold every ack back for a known time.
//
// No barrier at exit: the acks pop_front posts are what the sender's stop barrier waits on, and the
// durable read cursor is checkpointed by PrefetcherPipe::commit() when the object goes out of scope.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t hold_cycles = get_arg(args::hold_cycles);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);

    if constexpr (hold_cycles != 0) {
        riscv_wait(hold_cycles);
    }
    for (uint32_t i = 0; i < num_iters; ++i) {
        pipe.wait_front(1);
        pipe.pop_front(1, noc);
    }
}

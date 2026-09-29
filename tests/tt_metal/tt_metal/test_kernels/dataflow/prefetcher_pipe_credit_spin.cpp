// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe credit spinner: runs the pipe's credit protocol with no payload, to walk
// entries_sent / entries_acked up to a state a long-lived pipe reaches only after tens of
// gigabytes of traffic. Sender and receiver roles share this file and run concurrently.
//
// Compile-time parameters (via kernel compile_args):
//   [0] prefetcher_pipe_id
//   [1] num_ops         - reserve+push (sender) or wait+pop (receiver) iterations
//   [2] entries_per_op  - entries credited per iteration
//   [3] is_sender       - 1 for the producer role, 0 for the consumer role

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t prefetcher_pipe_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_ops = get_compile_time_arg_val(1);
    constexpr uint32_t entries_per_op = get_compile_time_arg_val(2);
    constexpr uint32_t is_sender = get_compile_time_arg_val(3);

    Noc noc;
    experimental::PrefetcherPipe pipe(prefetcher_pipe_id);

    if constexpr (is_sender != 0) {
        for (uint32_t i = 0; i < num_ops; ++i) {
            pipe.reserve_back(entries_per_op);
            // No write between reserve and push: the slots are re-credited untouched, which is
            // what makes the spin cheap. The cursor advances with the credits either way.
            pipe.push_back(entries_per_op, noc);
        }
    } else {
        for (uint32_t i = 0; i < num_ops; ++i) {
            pipe.wait_front(entries_per_op);
            pipe.pop_front(entries_per_op, noc);
        }
    }
}

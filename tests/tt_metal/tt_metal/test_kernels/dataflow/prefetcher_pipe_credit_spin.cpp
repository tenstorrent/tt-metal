// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Credit spin: run a PrefetcherPipe through num_ops * entries_per_op entries of credit protocol
// with no payload, to walk entries_sent / entries_acked up to a state a long-lived pipe reaches
// only after tens of gigabytes of traffic. Sender and receiver roles share this file (two
// KernelSpecs, one per role) and run concurrently.
//
// Bindings:
//   pipe::pipe             — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::num_ops          - reserve+push (sender) or wait+pop (receiver) iterations
//   args::entries_per_op   - entries credited per iteration
//   args::is_sender        - 1 for the producer role, 0 for the consumer role

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_ops = get_arg(args::num_ops);
    constexpr uint32_t entries_per_op = get_arg(args::entries_per_op);
    constexpr uint32_t is_sender = get_arg(args::is_sender);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::pipe);

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

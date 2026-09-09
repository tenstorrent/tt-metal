// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 DM driver for a PrefetcherPipe relayed through a DataflowBuffer declared in the
// ProgramSpec (DataflowBufferSpec::prefetcher_pipe_relays).
//
// The DFB is laid over the pipe rings, so its entries are the delivered bytes themselves and this
// kernel moves none of them: it turns a delivered entry into DFB credit for compute, and a compute
// pop back into a pipe ack.
//
// One relay DFB may span the receivers of several pipes, so the kernel cannot name a pipe id --
// which pipe a core belongs to varies across the cores running this same binary. It asks for the
// pipe by the relay DFB instead.
//
// Compile-time args (named):
//   num_entries - entries to consume

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    Noc noc;
    auto pipe = experimental::PrefetcherPipe::for_relay(dfb::relay);
    // Aligns the DFB to the pipe's durable cursor and arms pop_front's wait on compute. The
    // returned producer view is deliberately unused: publishing happens through the DFB below, and
    // publishing twice would hand compute twice the credit for the same bytes.
    pipe.bind_relay();
    DataflowBuffer relay(dfb::relay);

    // One entry of lookahead: publish the current entry, then let pop_front wait for compute to
    // finish the previous one before acking it back to the sender.
    for (uint32_t entry = 0; entry < num_entries; ++entry) {
        relay.reserve_back(1);
        pipe.wait_front(entry == 0 ? 1u : 2u);
        relay.push_back(1);
        if (entry >= 1) {
            pipe.pop_front(1, noc);
        }
    }
    if constexpr (num_entries > 0) {
        pipe.pop_front(1, noc);
    }
}

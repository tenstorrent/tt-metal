// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs on every node of the bound `target` DFB, so the DFB exists on all of them. The sender multicasts one entry of
// its DFB over the rectangle, so Watcher checks the rectangle against the DFB's nodes. With `read_ptr`, the destination
// is the DFB's read pointer, through read_ptr_view.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer target(dfb::target);
    Noc noc;
    if (!noc.is_local_bank(get_arg(args::sender_x), get_arg(args::sender_y))) {
        return;
    }
    target.reserve_back(1);
    const auto multicast = [&](const auto& destination) {
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(target.get_write_ptr()),
            destination,
            get_arg(args::size_bytes),
            get_arg(args::num_targets),
            {},
            {
                .noc_x_start = get_arg(args::noc_x_start),
                .noc_y_start = get_arg(args::noc_y_start),
                .noc_x_end = get_arg(args::noc_x_end),
                .noc_y_end = get_arg(args::noc_y_end),
            });
    };
    if constexpr (get_arg(args::read_ptr) != 0) {
        multicast(read_ptr_view(target));
    } else {
        multicast(target);
    }
    noc.async_write_barrier();
    target.push_back(1);
}

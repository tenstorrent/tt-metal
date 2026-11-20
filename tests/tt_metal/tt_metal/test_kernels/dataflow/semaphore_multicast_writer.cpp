// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicasts the bound `flag` semaphore over the rectangle, as `op` picks: 0 sets it, 1 increments it, 2 relays it into
// the bound `target` semaphore. Watcher checks the rectangle against the nodes of the semaphore written: `flag` for 0
// and 1, `target` for 2.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Semaphore flag(sem::flag);
    Semaphore target(sem::target);
    Noc noc;
    constexpr uint32_t op = get_arg(args::op);
    if constexpr (op == 0) {
        flag.set_multicast(
            noc,
            get_arg(args::noc_x_start),
            get_arg(args::noc_y_start),
            get_arg(args::noc_x_end),
            get_arg(args::noc_y_end),
            get_arg(args::num_targets));
        noc.async_write_barrier();
    } else if constexpr (op == 1) {
        flag.inc_multicast(
            noc,
            get_arg(args::noc_x_start),
            get_arg(args::noc_y_start),
            get_arg(args::noc_x_end),
            get_arg(args::noc_y_end),
            1,
            get_arg(args::num_targets));
        noc.async_atomic_barrier();
    } else {
        flag.relay_multicast(
            noc,
            target,
            get_arg(args::noc_x_start),
            get_arg(args::noc_y_start),
            get_arg(args::noc_x_end),
            get_arg(args::noc_y_end),
            get_arg(args::num_targets));
        noc.async_write_barrier();
    }
}

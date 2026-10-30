// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sets the bound `flag` semaphore over the rectangle, so Watcher checks the rectangle against the semaphore's nodes.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Semaphore flag(sem::flag);
    Noc noc;
    flag.set_multicast(
        noc,
        get_arg(args::noc_x_start),
        get_arg(args::noc_y_start),
        get_arg(args::noc_x_end),
        get_arg(args::noc_y_end),
        get_arg(args::num_targets));
    noc.async_write_barrier();
}

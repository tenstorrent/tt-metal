// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Holds a sub-range lock on a CoreLocalMem region while a writer on another core writes into it, so the
// NOC debugger's recorded extent can be checked against base + offset rather than base.

#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t region_base = get_arg(args::region_base);
    const uint32_t offset_elements = get_arg(args::offset_elements);
    const uint32_t num_elements = get_arg(args::num_elements);
    const uint32_t writer_noc_x = get_arg(args::writer_noc_x);
    const uint32_t writer_noc_y = get_arg(args::writer_noc_y);

    Noc noc;
    CoreLocalMem<uint32_t> buffer(region_base);
    Semaphore locked(sem::locked);    // the writer waits on its own instance; we remote-up it
    Semaphore written(sem::written);  // we wait on ours; the writer remote-ups it

    {
        // Lock [offset_elements, offset_elements + num_elements).
        auto lock = (buffer + offset_elements).scoped_lock(num_elements);

        locked.up(noc, writer_noc_x, writer_noc_y, 1);  // release the writer: the lock is held
        written.down(1);                                // hold it until the writer reports done
    }
}

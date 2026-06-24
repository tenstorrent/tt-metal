// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t writer_noc_x = get_arg(args::writer_noc_x);
    const uint32_t writer_noc_y = get_arg(args::writer_noc_y);
    const uint32_t local_scratch = get_arg(args::local_scratch);
    const uint32_t writer_inbox = get_arg(args::writer_inbox);

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    DataflowBuffer dfb(dfb::out);
    Semaphore locked(sem::locked);    // writer waits on its (writer-core) instance; we remote-up it
    Semaphore written(sem::written);  // we wait on our (this-core) instance; writer remote-ups it

    dfb.reserve_back(1);

    {
        auto lock = dfb.scoped_lock();  // whole ring locked for this scope

        // Publish the locked entry address to the writer (stage in local L1, then NOC it across).
        volatile tt_l1_ptr uint32_t* staged = (volatile tt_l1_ptr uint32_t*)(uintptr_t)local_scratch;
        *staged = dfb.get_write_ptr();
        CoreLocalMem<uint32_t> addr_src(local_scratch);
        noc.async_write(
            addr_src,
            unicast_endpoint,
            sizeof(uint32_t),
            {},
            {.noc_x = writer_noc_x, .noc_y = writer_noc_y, .addr = writer_inbox});
        noc.async_write_barrier();

        locked.up(noc, writer_noc_x, writer_noc_y, 1);  // release the writer (lock is held + addr is published)
        written.down(1);                                // hold the lock until the writer reports done
    }

    dfb.push_back(1);
}

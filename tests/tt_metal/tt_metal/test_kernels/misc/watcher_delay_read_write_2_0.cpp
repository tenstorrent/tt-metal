// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One NoC read (or, with WATCHER_DELAY_WRITE, one NoC write) to this core's own L1, so watcher debug delays fire.
// The writer waits for the reader so their non-atomic updates of the delay feedback word cannot overlap.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t local_addr = get_arg(args::local_addr);
    const uint32_t remote_addr = get_arg(args::remote_addr);
    const uint32_t size = get_arg(args::size);
    const uint32_t noc_x = get_arg(args::noc_x);
    const uint32_t noc_y = get_arg(args::noc_y);

    Noc noc;
    CoreLocalMem<std::uint32_t> local_buffer(local_addr);
    UnicastEndpoint endpoint;
    Semaphore read_done(sem::read_done);
#if defined(WATCHER_DELAY_WRITE)
    read_done.wait(1);
    noc.async_write(local_buffer, endpoint, size, {}, {.noc_x = noc_x, .noc_y = noc_y, .addr = remote_addr});
    noc.async_write_barrier();
#else
    noc.async_read(endpoint, local_buffer, size, {.noc_x = noc_x, .noc_y = noc_y, .addr = remote_addr}, {});
    noc.async_read_barrier();
    read_done.up(1);
#endif
}

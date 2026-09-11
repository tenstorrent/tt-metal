// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    const uint32_t write_size = get_arg(args::write_size);
    const uint32_t target_noc_x = get_arg(args::target_noc_x);  // the locker core
    const uint32_t target_noc_y = get_arg(args::target_noc_y);
    const uint32_t inbox = get_arg(args::inbox);  // local L1 word where the locker published the entry addr

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);
    Semaphore locked(sem::locked);
    Semaphore written(sem::written);

    locked.down(1);  // wait until the locker holds the lock and has published the target address
    const uint32_t target_addr = *(volatile tt_l1_ptr uint32_t*)(uintptr_t)inbox;
    noc.async_write(
        src_buffer,
        unicast_endpoint,
        write_size,
        {},
        {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr});
    noc.async_write_barrier();
    written.up(noc, target_noc_x, target_noc_y, 1);  // ack the locker so it releases the lock
}

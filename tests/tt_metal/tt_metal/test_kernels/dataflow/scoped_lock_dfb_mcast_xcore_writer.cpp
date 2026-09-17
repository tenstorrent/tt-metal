// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

// Cross-core multicast into another core's locked DFB ring. The mcast rectangle covers a row of cores with
// the locker in the interior (neither the start nor the end corner); this writer sits outside the rectangle.
// The tracker must iterate the whole rectangle -- not just the start corner -- to find the locker's locked
// DFB and flag WRITE_TO_LOCKED_DFB (src = this writer, dst = locker).
void kernel_main() {
    const uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    const uint32_t write_size = get_arg(args::write_size);
    const uint32_t locker_noc_x = get_arg(args::locker_noc_x);  // for the completion ack
    const uint32_t locker_noc_y = get_arg(args::locker_noc_y);
    const uint32_t mcast_noc_x_start = get_arg(args::mcast_noc_x_start);
    const uint32_t mcast_noc_y_start = get_arg(args::mcast_noc_y_start);
    const uint32_t mcast_noc_x_end = get_arg(args::mcast_noc_x_end);
    const uint32_t mcast_noc_y_end = get_arg(args::mcast_noc_y_end);
    const uint32_t num_dests = get_arg(args::num_dests);
    const uint32_t inbox = get_arg(args::inbox);  // local L1 word where the locker published the entry addr

    Noc noc;
    MulticastEndpoint mcast_endpoint;
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);
    Semaphore locked(sem::locked);
    Semaphore written(sem::written);

    locked.down(1);  // wait until the locker holds the lock and has published the target address
    const uint32_t target_addr = *(volatile tt_l1_ptr uint32_t*)(uintptr_t)inbox;
    noc.async_write_multicast(
        src_buffer,
        mcast_endpoint,
        write_size,
        num_dests,
        {},
        {.noc_x_start = mcast_noc_x_start,
         .noc_y_start = mcast_noc_y_start,
         .noc_x_end = mcast_noc_x_end,
         .noc_y_end = mcast_noc_y_end,
         .addr = target_addr});
    noc.async_write_barrier();
    written.up(noc, locker_noc_x, locker_noc_y, 1);  // ack the locker so it releases the lock
}

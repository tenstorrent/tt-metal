// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t is_leader = get_arg(args::is_leader);
    uint32_t my_x = get_arg(args::my_x);
    uint32_t my_y = get_arg(args::my_y);
    uint32_t leader_noc_x = get_arg(args::leader_noc_x);
    uint32_t leader_noc_y = get_arg(args::leader_noc_y);
    uint32_t expected_acks = get_arg(args::expected_acks);
    uint32_t sem_addr = get_arg(args::sem_addr);

    if (is_leader) {
        DEVICE_PRINT("[leader] ({},{}) waiting for {} acks\n", my_x, my_y, expected_acks);
        // Use uncached address for local polling so NOC writes are immediately visible.
        volatile tt_l1_ptr uint32_t* sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr + MEM_L1_UNCACHED_BASE);
        noc_semaphore_wait(sem_ptr, expected_acks);
        DEVICE_PRINT("[leader] got {} acks\n", *sem_ptr);
    } else {
        DEVICE_PRINT("[worker] hello from ({},{})\n", my_x, my_y);
        Noc noc;
        // sem_addr is the physical L1 offset (no uncached base) for the NOC address.
        uint64_t noc_addr = get_noc_addr(leader_noc_x, leader_noc_y, sem_addr);
        noc_semaphore_inc(noc_addr, 1, noc.get_noc_id());
        noc.async_atomic_barrier();
    }
}

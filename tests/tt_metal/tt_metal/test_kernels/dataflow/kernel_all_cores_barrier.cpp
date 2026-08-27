// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#endif

void kernel_main() {
    const uint32_t arrivals_addr = get_arg_val<uint32_t>(0);
    const uint32_t observed_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t rounds = get_arg_val<uint32_t>(3);
    const uint32_t skew_iters = get_arg_val<uint32_t>(4);
    const uint32_t max_participants = get_arg_val<uint32_t>(5);

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t participant = thread_id;

    volatile tt_l1_ptr uint32_t* arrivals =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals_addr + MEM_L1_UNCACHED_BASE);
    volatile tt_l1_ptr uint32_t* post =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(post_addr + MEM_L1_UNCACHED_BASE);
    (void)observed_addr;

    sync_all_cores(max_participants);

    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t delay = (participant + 1) * skew_iters;
        for (uint32_t d = 0; d < delay; d++) {
            asm volatile("nop");
        }

        arrivals[r * max_participants + participant] = 1;
        sync_all_cores(max_participants);

        sync_all_cores(max_participants);

        post[r * max_participants + participant] = 1;
        sync_all_cores(max_participants);
    }
}

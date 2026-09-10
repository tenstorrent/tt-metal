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
    const uint32_t num_dm_threads = get_arg_val<uint32_t>(6);
    const uint32_t num_tensixes = get_arg_val<uint32_t>(7);

    // DM harts occupy the low slots of the shared arrival array; the compute kernel takes the rest.
    const uint32_t participant = get_my_thread_id();

    volatile tt_l1_ptr uint32_t* arrivals =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals_addr + MEM_L1_UNCACHED_BASE);
    volatile tt_l1_ptr uint32_t* post =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(post_addr + MEM_L1_UNCACHED_BASE);
    (void)observed_addr;  // the compute kernel's TRISC 0 is the observer

    dm_compute_barrier(num_dm_threads, num_tensixes);

    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t delay = (participant + 1) * skew_iters;
        for (uint32_t d = 0; d < delay; d++) {
            asm volatile("nop");
        }

        arrivals[r * max_participants + participant] = 1;
        dm_compute_barrier(num_dm_threads, num_tensixes);

        // Matches the observer's barrier in the compute kernel, so both sides stay in lockstep.
        dm_compute_barrier(num_dm_threads, num_tensixes);

        post[r * max_participants + participant] = 1;
        dm_compute_barrier(num_dm_threads, num_tensixes);
    }
}

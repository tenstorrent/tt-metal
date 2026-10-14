// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program A, NCRISC on the "A cores". Waits until the flood is established, issues N_INCS non-posted
// atomic increments to the far core, and returns. With A_BARRIER=0 it does not wait for the atomic
// responses (the reader_worker_dispatch.cpp pattern); with A_BARRIER=1 it waits (control) and records
// how long the response took.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "late_atomic_common.h"

void kernel_main() {
    constexpr uint32_t A_BARRIER = get_compile_time_arg_val(0);
    constexpr uint32_t N_INCS = get_compile_time_arg_val(1);
    constexpr uint32_t DELAY = get_compile_time_arg_val(2);
    constexpr uint32_t JITTER = get_compile_time_arg_val(3);
    constexpr uint32_t LATE_THRESH = get_compile_time_arg_val(4);

    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t far_x = get_arg_val<uint32_t>(1);
    const uint32_t far_y = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    const uint32_t run = st[S_A_RUNS] + 1;
    st[S_A_RUNS] = run;

    // Vary the firing point inside the flood from run to run (works under trace replay too).
    uint32_t wait = DELAY;
    if constexpr (JITTER > 0) {
        wait += mix32(run ^ (stats_addr << 7) ^ (far_x << 3)) % JITTER;
    }
    spin_cycles(wait);

    const uint64_t far_sem = get_noc_addr(far_x, far_y, stats_addr + S_SEM_FAR * sizeof(uint32_t));
    for (uint32_t i = 0; i < N_INCS; ++i) {
        noc_semaphore_inc(far_sem, 1);
    }

    if constexpr (A_BARRIER) {
        const uint32_t t0 = c_tensix_core::read_wall_clock_l();
        noc_async_atomic_barrier();
        const uint32_t dt = c_tensix_core::read_wall_clock_l() - t0;
        if (dt > st[S_A_BAR_MAX]) {
            st[S_A_BAR_MAX] = dt;
        }
        st[S_A_BAR_SUM16] = st[S_A_BAR_SUM16] + (dt >> 4);
        if (dt > LATE_THRESH) {
            st[S_A_BAR_LATE] = st[S_A_BAR_LATE] + 1;
        }
    }
    st[S_A_EXIT_TS] = c_tensix_core::read_wall_clock_l();
}

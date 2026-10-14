// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program B, NCRISC on the same cores and NoC as program A.
//
// The NCRISC kernel prologue (ncrisck.cc) snapshots NIU_MST_ATOMIC_RESP_RECEIVED into
// noc_nonposted_atomics_acked before wait_for_go_message(). Any response to A's atomic that lands
// after that snapshot leaves hw == sw + 1. This kernel first measures that difference (probe), then
// runs one barrier pattern:
//   0 probe         measure only
//   1 inc_barrier   1 inc to self, noc_async_atomic_barrier()          (own response still in flight)
//   2 incn_barrier  B_INCS incs round-robin over the peer cores, barrier (moe_fused_swiglu invite loop)
//   3 late_barrier  1 inc to self, wait PROBE_WAIT, barrier             (own response already back)
//   4 full_barrier  noc_async_full_barrier() with nothing outstanding

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "late_atomic_common.h"

FORCE_INLINE uint32_t atomic_hw() { return NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED); }
FORCE_INLINE uint32_t atomic_sw() { return noc_nonposted_atomics_acked[noc_index]; }

void kernel_main() {
    const uint32_t hw0 = atomic_hw();
    const uint32_t sw0 = atomic_sw();
    const uint32_t t_start = c_tensix_core::read_wall_clock_l();

    constexpr uint32_t B_MODE = get_compile_time_arg_val(0);
    constexpr uint32_t PROBE_WAIT = get_compile_time_arg_val(1);
    constexpr uint32_t B_INCS = get_compile_time_arg_val(2);

    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t self_x = get_arg_val<uint32_t>(1);
    const uint32_t self_y = get_arg_val<uint32_t>(2);
    const uint32_t n_peers = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = stats_addr + S_SEM_PEER * sizeof(uint32_t);

    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    const uint32_t run = st[S_B_RUNS] + 1;
    st[S_B_RUNS] = run;

    const uint32_t gap = t_start - st[S_A_EXIT_TS];
    if (run == 1 || gap < st[S_B_GAP_MIN]) {
        st[S_B_GAP_MIN] = gap;
    }
    if (gap > st[S_B_GAP_MAX]) {
        st[S_B_GAP_MAX] = gap;
    }

    spin_cycles(PROBE_WAIT);
    const uint32_t hw1 = atomic_hw();
    const uint32_t sw1 = atomic_sw();
    const uint32_t d0 = hw0 - sw0;
    const uint32_t d1 = hw1 - sw1;
    if (d0 != 0) {
        st[S_B_D0_NZ] = st[S_B_D0_NZ] + 1;
    }
    if (d1 != 0) {
        if (st[S_B_D1_NZ] == 0) {
            st[S_B_FIRST_HIT] = run;
        }
        st[S_B_D1_NZ] = st[S_B_D1_NZ] + 1;
        st[S_B_LAST_D1] = d1;
    }
    const uint32_t dmax = d0 > d1 ? d0 : d1;
    if (dmax > st[S_B_DMAX]) {
        st[S_B_DMAX] = dmax;
    }

    st[S_B_ENTRY_HW] = hw1;
    st[S_B_ENTRY_SW] = sw1;
    st[S_B_STATE] = B_STATE_IN_ACTION;
    const uint32_t t_action = c_tensix_core::read_wall_clock_l();

    if constexpr (B_MODE == 1) {
        noc_semaphore_inc(get_noc_addr(self_x, self_y, sem_addr), 1);
        noc_async_atomic_barrier();
    } else if constexpr (B_MODE == 2) {
        for (uint32_t i = 0; i < B_INCS; ++i) {
            const uint32_t p = i % n_peers;
            const uint32_t px = get_arg_val<uint32_t>(4 + 2 * p);
            const uint32_t py = get_arg_val<uint32_t>(5 + 2 * p);
            noc_semaphore_inc(get_noc_addr(px, py, sem_addr), 1);
        }
        noc_async_atomic_barrier();
    } else if constexpr (B_MODE == 3) {
        noc_semaphore_inc(get_noc_addr(self_x, self_y, sem_addr), 1);
        spin_cycles(PROBE_WAIT);
        noc_async_atomic_barrier();
    } else if constexpr (B_MODE == 4) {
        noc_async_full_barrier();
    }

    const uint32_t action = c_tensix_core::read_wall_clock_l() - t_action;
    if (action > st[S_B_ACTION_MAX]) {
        st[S_B_ACTION_MAX] = action;
    }
    st[S_B_STATE] = B_STATE_DONE;

    if constexpr (B_MODE != 0) {
        spin_cycles(PROBE_WAIT);
        if (atomic_hw() != atomic_sw()) {
            st[S_B_RESID_NZ] = st[S_B_RESID_NZ] + 1;
        }
    }
}

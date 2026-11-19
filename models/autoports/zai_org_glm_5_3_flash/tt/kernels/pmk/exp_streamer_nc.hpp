// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK program B, PMK_MCAST_DUAL (moe2/PERF.md sec 10): the streamer core's NCRISC (NoC1) multicasts the SECOND half of
// every chunk the BRISC (exp_streamer.hpp) hands over through the sync words -- GO (n, slot, receivers' dst, bytes,
// DOWN = the ring kind) after the BRISC waited the rectangle's credits -- then SETs the ring's second VALID (semaphore
// 12 gate|up / 13 down) on the receivers behind its data (same NoC, ordered), flushes, and reports DONE = n (the slot
// half left L1). A GO with bytes 0 ends the launch. NoC1 multicasts go high -> low corner (ttnn's convention;
// kernels/pfs). The VALID source words are the 16-B-aligned SW_VAL1_GU / SW_VAL1_D of the sync shard (perf5: the
// 22:23Z watcher NOC sanitize abort was a 4-B-aligned source at SW_DONE + 4); the ring kind comes from SW_GO_DOWN
// (perf5: the byte-count heuristic was wrong wherever GU_BB == DN_BB, e.g. (1, 4, 8)).
// Runtime args: the same list as the BRISC (RT0 + 4 sync, + 5..8 the box, + 9 P_compute, + 13 g).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

using namespace pmk_moe;

namespace expsnc {
constexpr uint32_t SW_GO = 96, SW_DONE = 112, SW_VAL1_GU = 128, SW_VAL1_D = 144;
constexpr uint32_t SEM_VALID1_GU = 12, SEM_VALID1_D = 13;
static_assert(SW_VAL1_GU % 16 == 0 && SW_VAL1_D % 16 == 0, "multicast source words are 16-B aligned");
}  // namespace expsnc

FORCE_INLINE void phase_main() {
    const uint32_t sync = get_arg_val<uint32_t>(RT0 + 4);
    const uint32_t lx = get_arg_val<uint32_t>(RT0 + 5), ly = get_arg_val<uint32_t>(RT0 + 6);
    const uint32_t hx = get_arg_val<uint32_t>(RT0 + 7), hy = get_arg_val<uint32_t>(RT0 + 8);
    const uint32_t p_compute = get_arg_val<uint32_t>(RT0 + 9);
    const uint32_t g = get_arg_val<uint32_t>(RT0 + 13);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + expsnc::SW_GO);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + expsnc::SW_DONE);
    volatile tt_l1_ptr uint32_t* val_gu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + expsnc::SW_VAL1_GU);
    volatile tt_l1_ptr uint32_t* val_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + expsnc::SW_VAL1_D);
    // the BRISC zeroes the sync words at its start; the plan barrier (ms later) orders this RISC behind that
    wait_min(SEM_PLAN3, p_compute);
    const uint32_t vgu = get_semaphore(expsnc::SEM_VALID1_GU), vd = get_semaphore(expsnc::SEM_VALID1_D);
    uint32_t n = 0, n_gu = 0, n_d = 0;
    while (true) {
        while (go[0] <= n) {
            invalidate_l1_cache();
        }
        ++n;
        const uint32_t slot = go[1], dst = go[2], bytes = go[3], down = go[5];
        if (bytes == 0) {
            break;
        }
        noc_async_write_multicast(slot, get_noc_multicast_addr(hx, hy, lx, ly, dst), bytes, g, false);
        // the VALID count behind the half (same NoC, ordered); the source word is consumed before the next GO's
        // write to it: noc_async_writes_flushed below
        volatile tt_l1_ptr uint32_t* val = down ? val_d : val_gu;
        *val = down ? ++n_d : ++n_gu;
        noc_semaphore_set_multicast(
            reinterpret_cast<uint32_t>(val), get_noc_multicast_addr(hx, hy, lx, ly, down ? vd : vgu), g, false);
        noc_async_writes_flushed();
        *done = n;
    }
    noc_async_write_barrier();
    *done = n;
}

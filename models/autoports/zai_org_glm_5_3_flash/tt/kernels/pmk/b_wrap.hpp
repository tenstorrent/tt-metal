// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PROGRAM B AS A WINDOW OF THE FUSED LAYER PROGRAM (Stage 16 M-B, 2026-10-24; tt/pmk_b_window.py). Program B's own
// kernels (moe_brisc / moe_ncrisc / moe_trisc on the receivers, moe_stream_brisc / _ncrisc on the streamers) run as
// one phase of the family's composites, each wrapped (b_wrap_*.cpp) behind GUARD words at the part's ph_rt0:
//   DM:    [active, hub_x, hub_y, is_hub, base, n]   (B's own args follow: its PMK_RT_* defines are ph_rt0 + 6 + k)
//   TRISC: [active]                                 (ph_rt0 + 1 + k)
// active = 0 on a composite core that is not one of B's (a member composite core is a receiver; a root / MM-only
// core a streamer only if it is one of B's 24).
// THE A -> B SEMAPHORE HAND-OVER: B's ids 5 .. 13 carry other meanings in program A (and A's late credits may still be
// in flight to some core). Two program-wide rounds on semaphore PMK_BW_RDV (14 = FS_REDUCED: A's H6 counter, whose
// value after A is known on the host: `base`) between B's n cores, hub = the first receiver:
//   round 1: every core's BRISC (after its window entry barriered the core's NoC traffic) +1 the hub; the hub waits
//            base + n and +1 every B core (B's own receiver / streamer lists) -> no A traffic is in flight anywhere;
//   reset:   every core zeroes ITS ids 5 .. 13;
//   round 2: as round 1 (the hub waits base + 2 n + 1) -> every core has reset before any core's B traffic starts.
// The NCRISCs wait the round-2 release on their own core before B (their first B wait is on a reset id).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef PMK_BW_RDV
#define PMK_BW_RDV 14
#endif

namespace pmk_bw {

constexpr uint32_t GW_DM = 6, GW_C = 1;
constexpr uint32_t SEM_LO = 5, SEM_HI = 13;

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

// the hub's release: +1 on PMK_BW_RDV of every B core = the P receivers (B's [x, y] x P list at cores_arg) and the
// streamers (B's [n, (x, y) x n] list at st_arg)
FORCE_INLINE void release_all(uint32_t cores_arg, uint32_t p, uint32_t st_arg) {
    for (uint32_t r = 0; r < p; ++r) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(cores_arg + 2 * r),
                get_arg_val<uint32_t>(cores_arg + 2 * r + 1),
                get_semaphore(PMK_BW_RDV)),
            1);
    }
    const uint32_t ns = get_arg_val<uint32_t>(st_arg);
    for (uint32_t r = 0; r < ns; ++r) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(st_arg + 1 + 2 * r),
                get_arg_val<uint32_t>(st_arg + 2 + 2 * r),
                get_semaphore(PMK_BW_RDV)),
            1);
    }
    noc_async_atomic_barrier();
}

// the BRISC's two rounds (HUB = the receiver composite: the only one with B's core lists)
template <bool HUB>
FORCE_INLINE void handover(uint32_t g, uint32_t cores_arg, uint32_t p, uint32_t st_arg) {
    const uint32_t hx = get_arg_val<uint32_t>(g + 1), hy = get_arg_val<uint32_t>(g + 2);
    const uint32_t hub = get_arg_val<uint32_t>(g + 3), base = get_arg_val<uint32_t>(g + 4),
                   n = get_arg_val<uint32_t>(g + 5);
    volatile tt_l1_ptr uint32_t* s = sem(PMK_BW_RDV);
    const uint64_t hs = get_noc_addr(hx, hy, get_semaphore(PMK_BW_RDV));
    noc_semaphore_inc(hs, 1);
    noc_async_atomic_barrier();
    if (HUB && hub) {
        noc_semaphore_wait_min(s, base + n);
        release_all(cores_arg, p, st_arg);
    } else {
        noc_semaphore_wait_min(s, base + 1);
    }
    for (uint32_t id = SEM_LO; id <= SEM_HI; ++id) {
        *sem(id) = 0;
    }
    noc_semaphore_inc(hs, 1);
    noc_async_atomic_barrier();
    if (HUB && hub) {
        noc_semaphore_wait_min(s, base + 2 * n + 1);
        release_all(cores_arg, p, st_arg);
    } else {
        noc_semaphore_wait_min(s, base + 2);
    }
}

// THE STREAMER HUB (mb8, TP8): at TP8 H6's aggregator SETs the members' and roots' id 14 (pmk_agg.cpp) at a time no B
// core knows -- an MM-only streamer leaves A after P8 and its round-1 +1 could reach a member hub BEFORE that SET,
// which would erase it (the hub never reaches base + n). So the hub is a streamer whose id 14 only counts (MM-only /
// tail: base 0). Streamer guard: [active, hx, hy, is_hub, base, n, rel_off] (GW_S = 7); the hub's release list [x, y] x
// n (every B core, itself included, as release_all) sits at ph_rt0 + rel_off.
constexpr uint32_t GW_S = 7;
FORCE_INLINE void release_list(uint32_t list_arg, uint32_t n) {
    for (uint32_t r = 0; r < n; ++r) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(list_arg + 2 * r),
                get_arg_val<uint32_t>(list_arg + 2 * r + 1),
                get_semaphore(PMK_BW_RDV)),
            1);
    }
    noc_async_atomic_barrier();
}

// the streamer BRISC's two rounds (the same protocol as handover; the hub releases from its own list)
FORCE_INLINE void handover_s(uint32_t g) {
    const uint32_t hx = get_arg_val<uint32_t>(g + 1), hy = get_arg_val<uint32_t>(g + 2);
    const uint32_t hub = get_arg_val<uint32_t>(g + 3), base = get_arg_val<uint32_t>(g + 4),
                   n = get_arg_val<uint32_t>(g + 5);
    const uint32_t rel = g + get_arg_val<uint32_t>(g + 6);
    volatile tt_l1_ptr uint32_t* s = sem(PMK_BW_RDV);
    const uint64_t hs = get_noc_addr(hx, hy, get_semaphore(PMK_BW_RDV));
    noc_semaphore_inc(hs, 1);
    noc_async_atomic_barrier();
    if (hub) {
        noc_semaphore_wait_min(s, base + n);
        release_list(rel, n);
    } else {
        noc_semaphore_wait_min(s, base + 1);
    }
    for (uint32_t id = SEM_LO; id <= SEM_HI; ++id) {
        *sem(id) = 0;
    }
    noc_semaphore_inc(hs, 1);
    noc_async_atomic_barrier();
    if (hub) {
        noc_semaphore_wait_min(s, base + 2 * n + 1);
        release_list(rel, n);
    } else {
        noc_semaphore_wait_min(s, base + 2);
    }
}

// the NCRISC: B starts after this core's round-2 release (the hub's own release is its last +1)
FORCE_INLINE void wait_go(uint32_t g) {
    const uint32_t hub = get_arg_val<uint32_t>(g + 3), base = get_arg_val<uint32_t>(g + 4),
                   n = get_arg_val<uint32_t>(g + 5);
    noc_semaphore_wait_min(sem(PMK_BW_RDV), base + (hub ? 2 * n + 2 : 2));
}

}  // namespace pmk_bw

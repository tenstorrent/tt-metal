// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The per-launch captures of the tile-wide counters: the NoC interface counters and the NoC router counters. The
// firmware opens them before it starts the cores of a launch; the core that closes them holds back room for their
// markers at its start and waits for the other cores at its end. Each compiles to nothing unless its option is on.

#pragma once

#include "internal/profiler/noc_counters.hpp"
#include "internal/profiler/router_counters.hpp"

#define TileCountersBegin()    \
    do {                       \
        NocCountersBegin();    \
        RouterCountersBegin(); \
    } while (0)

#define TileCountersReserve()    \
    do {                         \
        NocCountersReserve();    \
        RouterCountersReserve(); \
    } while (0)

#if ((defined(NOC_COUNTERS_CAPTURE) && defined(NOC_COUNTERS_CLOSE)) || \
     (defined(ROUTER_COUNTERS_CAPTURE) && defined(ROUTER_COUNTERS_CLOSE))) && \
    !defined(ARCH_QUASAR)
// The window spans the launch, so the BRISC kernel waits for NCRISC and the TRISCs as the firmware does after it.
inline __attribute__((always_inline)) void tile_counters_wait_for_launch() {
    auto* sync = reinterpret_cast<volatile tt_l1_ptr subordinate_map_t*>(GET_MAILBOX_ADDRESS_DEV(subordinate_sync.map));
    while (sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
#if defined(ARCH_WORMHOLE)
        asm volatile("nop; nop; nop; nop; nop");
#endif
        invalidate_l1_cache();
    }
}
#define TileCountersWaitForLaunch() tile_counters_wait_for_launch()
#else
#define TileCountersWaitForLaunch()
#endif

#define TileCountersEnd()            \
    do {                             \
        TileCountersWaitForLaunch(); \
        RouterCountersEnd();         \
        NocCountersEnd();            \
    } while (0)

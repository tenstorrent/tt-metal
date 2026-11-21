// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The per-launch captures of the tile-wide counters, which one core of the tile runs around the launch of all its
// cores: the NoC interface counters and the NoC router counters. Each compiles to nothing unless its option is on.

#pragma once

#include "internal/profiler/noc_counters.hpp"
#include "internal/profiler/router_counters.hpp"

#define TileCountersBegin()    \
    do {                       \
        NocCountersBegin();    \
        RouterCountersBegin(); \
    } while (0)

#if (defined(NOC_COUNTERS_CAPTURE) || defined(ROUTER_COUNTERS_CAPTURE)) && !defined(ARCH_QUASAR)
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

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of watcher_waypoints.cpp.
// Compiled only for TENSIX cores (BRISC / NCRISC / TRISC / DM). Active and idle
// ethernet callers continue to use watcher_waypoints.cpp via the legacy host API.

#include <cstdint>
#include "api/debug/waypoint.h"
#include "risc_common.h"
#include "experimental/kernel_args.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    WAYPOINT("AAAA");

    uint32_t sync_flag_addr = get_arg(args::sync_flag_addr);
    volatile uint32_t* sync_flag = reinterpret_cast<volatile uint32_t*>(sync_flag_addr);

    while (*sync_flag != 1) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // On Quasar DM cores, we must invalidate L2 to see updated host writes in L1.
        invalidate_l2_cache_line(static_cast<uintptr_t>(sync_flag_addr));
#else
        invalidate_l1_cache();
#endif
    }
}

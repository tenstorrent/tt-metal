// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grid multicast fan-out (source-side). Stages `value` in local L1, then NoC-multicasts it to
// `result_addr` on every node in the [x_start,y_start]-[x_end,y_end] physical rectangle. This
// drives the actual NoC multicast path (get_noc_multicast_addr -> for_each_multicast_coordinate),
// so a dropped row/column shows up as an un-updated node in the host readback. num_dests excludes
// the source (plain multicast); the source covers its own slot via the local write above.
// Non-DFB: pure NoC data multicast.

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t value = get_arg(args::value);
    const uint32_t result_addr = get_arg(args::result_addr);
    const uint32_t x_start = get_arg(args::mcast_x_start);
    const uint32_t y_start = get_arg(args::mcast_y_start);
    const uint32_t x_end = get_arg(args::mcast_x_end);
    const uint32_t y_end = get_arg(args::mcast_y_end);
    const uint32_t num_dests = get_arg(args::num_dests);

    // Stage the value locally (covers the source's own slot).
    CoreLocalMem<uint32_t> buf(result_addr);
    buf[0] = value;
    flush_l2_cache_line(result_addr);

    // Multicast it to result_addr on every OTHER node in the rectangle.
    const uint64_t mcast = get_noc_multicast_addr(x_start, y_start, x_end, y_end, result_addr);
    noc_async_write_multicast(result_addr, mcast, sizeof(uint32_t), num_dests);
    noc_async_write_barrier();
}

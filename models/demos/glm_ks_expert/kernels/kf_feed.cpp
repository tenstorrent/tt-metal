// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Feeds the real flat gate_up_compute.cpp on one core (BRISC): the schedule page (CB 6), then x blocks (CB 0, the x
// ring) and weight blocks (CB 1, the weight ring) as fast as their rings free up - the data already in L1 (whatever
// the rings hold: perf only). RT: 0 entries, 1 sub-blocks an entry, 2 x blocks a sub-block, 3 x block tiles, 4 weight
// blocks a load, 5 weight block tiles, 6 row tiles of a full sub-block
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ne = get_arg_val<uint32_t>(0), sbe = get_arg_val<uint32_t>(1), nb = get_arg_val<uint32_t>(2);
    const uint32_t xblk = get_arg_val<uint32_t>(3), nw = get_arg_val<uint32_t>(4), wblk = get_arg_val<uint32_t>(5);
    const uint32_t mt = get_arg_val<uint32_t>(6);
    constexpr uint32_t kMax = 2 * get_named_compile_time_arg_val("max_experts");
    cb_reserve_back(6, 1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(6));
    page[0] = ne;
    page[1] = ne * sbe;
    for (uint32_t a = 0; a < ne; ++a) {
        page[2 + a] = sbe;
        page[2 + kMax + a] = a | ((a % 2) << 8) | (1u << 16);
        page[2 + 2 * kMax + a] = mt;
    }
    cb_push_back(6, 1);
    const uint32_t x_total = ne * sbe * nb, w_total = ne * nw;
    uint32_t xs = 0, wsent = 0;
    while (xs < x_total || wsent < w_total) {
        if (wsent < w_total && cb_pages_reservable_at_back(1, wblk)) {
            cb_push_back(1, wblk);
            ++wsent;
        }
        if (xs < x_total && cb_pages_reservable_at_back(0, xblk)) {
            cb_push_back(0, xblk);
            ++xs;
        }
    }
}

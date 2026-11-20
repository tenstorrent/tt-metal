// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hub of the Hy4 decode MoE program (BRISC, NOC1), given routing (fused/moe/program.py): the 8 token rows x
// ([ROWS, 32] BF16 row tiles, in this core's L1) and the routing result ([U, ids | w], in this core's L1) go to every
// worker core of the chip in two multicasts, then one multicast semaphore increment releases them.
//
// runtime args: [x_addr, meta_in_addr, status_addr]
// compile-time args: [cb_x, cb_meta, x_bytes, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "moe_dm.hpp"

void kernel_main() {
#ifdef HY4_LAUNCH_ONLY  // measurement only: what launching this program costs
    return;
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t meta_in_addr = get_arg_val<uint32_t>(1);
    const uint32_t status_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(3);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(4);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(6);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(7);

    const uint32_t x_l1 = get_write_ptr(cb_x);
    const uint32_t meta_l1 = get_write_ptr(cb_meta);

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_addr);
    const uint32_t runs = status[hy4moe::ST_RUNS];
    for (uint32_t i = 0; i < hy4moe::ST_WORDS; ++i) {
        status[i] = 0;
    }
    status[hy4moe::ST_RUNS] = runs + 1;
    // the down cores count themselves here at the end; whoever consumes the output takes the count from zero
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hy4moe::sem(hy4moe::SEM_OUT))[0] = 0;
#ifdef HY4_TIMING
    const uint32_t t0 = hy4moe::now();
#endif

    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    volatile tt_l1_ptr uint32_t* meta_in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_in_addr);
    for (uint32_t i = 0; i < hy4moe::META_IN_BYTES / 4; ++i) {
        meta[i] = meta_in[i];
    }
    for (uint32_t i = hy4moe::META_IN_BYTES / 4; i < hy4moe::META_BYTES / 4; ++i) {
        meta[i] = 0;
    }

    // Every NOC wait is bounded: a hub that cannot send says so in the status words and ends; the other cores then
    // give up on their own bounded wait for the release.
    uint32_t failed = 0;
    if (!hy4moe::grid_mcast_write(x_addr, mc_x0, mc_y0, mc_x1, mc_y1, x_l1, x_bytes, mc_dests)) {
        failed = 1;
    } else if (!hy4moe::grid_mcast_write(meta_l1, mc_x0, mc_y0, mc_x1, mc_y1, meta_l1, hy4moe::META_BYTES, mc_dests)) {
        failed = 2;
    } else if (!hy4moe::writes_flushed()) {
        failed = 4;
    } else if (!hy4moe::grid_mcast_inc(mc_x0, mc_y0, mc_x1, mc_y1, hy4moe::sem(hy4moe::SEM_GO), 1, mc_dests)) {
        failed = 3;
    } else if (!hy4moe::atomics_flushed()) {
        failed = 4;
    }
    status[hy4moe::ST_HUB] = failed;
#ifdef HY4_TIMING
    status[hy4moe::ST_HUB_TIMING + 2] = hy4moe::now() - t0;
#endif
}

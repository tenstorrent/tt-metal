// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hub of the Hy4 decode projection programs (BRISC, NOC1) (fused/projections/program.py), derived from
// fused/moe/kernels/hub.cpp: the input buffer ([ROWS, 32] BF16 row tiles, in this core's L1) and the parameter page
// go to every worker core of the chip in two multicasts, then one multicast increment of the go word releases them.
// In a layer program this role is the landing of the cross-chip step that brings the input.
//
// runtime args: [x_addr, meta_in_addr, status_addr, go_addr]
// compile-time args: [cb_x, cb_meta, x_bytes, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests, done_words]
//                     done_words: the units' done counters, words 1 .. done_words of the go tensor (1 unless the
//                     program has several done groups)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "proj_dm.hpp"

void kernel_main() {
#ifdef HY4_EMPTY  // measurement only: the launch floor of the program
    return;
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t meta_in_addr = get_arg_val<uint32_t>(1);
    const uint32_t status_addr = get_arg_val<uint32_t>(2);
    const uint32_t go_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(3);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(4);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(6);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(7);
    constexpr uint32_t done_words = get_compile_time_arg_val(8);

    const uint32_t x_l1 = get_write_ptr(cb_x);
    const uint32_t meta_l1 = get_write_ptr(cb_meta);

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_addr);
    const uint32_t runs = status[hy4proj::ST_RUNS];
    for (uint32_t i = 0; i < hy4proj::ST_WORDS; ++i) {
        status[i] = 0;
    }
    status[hy4proj::ST_RUNS] = runs + 1;
    // units that have written their tiles into the output buffer (they increment it; proj_writer.cpp)
    for (uint32_t i = 1; i <= done_words; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr)[i] = 0;
    }
#ifdef HY4_TIMING
    const uint32_t t0 = hy4proj::now();
#endif

    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    volatile tt_l1_ptr uint32_t* meta_in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_in_addr);
    for (uint32_t i = 0; i < hy4proj::META_IN_BYTES / 4; ++i) {
        meta[i] = meta_in[i];
    }
    for (uint32_t i = hy4proj::META_IN_BYTES / 4; i < hy4proj::META_BYTES / 4; ++i) {
        meta[i] = 0;
    }

    // Every NOC wait is bounded: a hub that cannot send says so in the status words and ends; the units then give
    // up on their own bounded wait for the release.
    uint32_t failed = 0;
    if (!hy4proj::grid_mcast_write(x_addr, mc_x0, mc_y0, mc_x1, mc_y1, x_l1, x_bytes, mc_dests)) {
        failed = 1;
    } else if (!hy4proj::grid_mcast_write(
                   meta_l1, mc_x0, mc_y0, mc_x1, mc_y1, meta_l1, hy4proj::META_BYTES, mc_dests)) {
        failed = 2;
    } else if (!hy4proj::writes_flushed()) {
        failed = 4;
    } else if (!hy4proj::grid_mcast_inc(mc_x0, mc_y0, mc_x1, mc_y1, go_addr, mc_dests)) {
        failed = 3;
    } else if (!hy4proj::atomics_flushed()) {
        failed = 4;
    }
    status[hy4proj::ST_HUB] = failed;
#ifdef HY4_TIMING
    status[hy4proj::ST_HUB_TIMING] = hy4proj::now() - t0;
#endif
}

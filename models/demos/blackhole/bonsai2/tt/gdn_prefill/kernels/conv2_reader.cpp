// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv (conv2_compute.cpp), reader (NCRISC). Per owned channel tile-column: the conv
// state tile, the column's tap-vector tile, then the input tiles (tt, ct) of x in time order (8 per read barrier).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: x, state, tap vectors, consts, Dt, ncores, grid_x, Tt, Xw (x width in tiles)
    const uint32_t Dt = get_common_arg_val<uint32_t>(4);
    const uint32_t Tt = get_common_arg_val<uint32_t>(7), Xw = get_common_arg_val<uint32_t>(8);
    uint32_t first, n;
    core_span(Dt * Tt, get_common_arg_val<uint32_t>(5), get_common_arg_val<uint32_t>(6), 1, first, n);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ws = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), 2048);
    constexpr uint32_t cb_x = 0, cb_s = 1, cb_w = 2, cb_c = 3, cb_p = 6, cb_l = 7, NCONST = 11, BATCH = 4;
    cb_reserve_back(cb_c, NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc_async_read(cs.get_noc_addr(i), get_write_ptr(cb_c) + i * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, NCONST);
    const uint64_t me = get_noc_addr(0);  // this core's L1 (local copies into c_6)
    uint32_t pushed = 0;                  // tiles pushed to c_0 (8-page ring): batches never cross a multiple of 4
    for (uint32_t u = first; u < first + n;) {
        const uint32_t ct = u / Tt, tt0 = u % Tt;
        const uint32_t seg = (Tt - tt0) < (first + n - u) ? (Tt - tt0) : (first + n - u);
        cb_reserve_back(cb_w, 1);
        noc_async_read(ws.get_noc_addr(ct), get_write_ptr(cb_w), 2048);
        if (tt0 == 0) {
            // the column's old state, and its last input tile for the new state (the core reading the old state
            // writes the new one: no other core can overwrite it first)
            cb_reserve_back(cb_s, 1);
            noc_async_read(ss.get_noc_addr(ct), get_write_ptr(cb_s), 2048);
            cb_reserve_back(cb_l, 1);
            noc_async_read(xs.get_noc_addr((Tt - 1) * Xw + ct), get_write_ptr(cb_l), 2048);
        } else {  // a segment from inside the column: its first tile's previous tile
            cb_reserve_back(cb_p, 1);
            noc_async_read(xs.get_noc_addr((tt0 - 1) * Xw + ct), get_write_ptr(cb_p), 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
        if (tt0 == 0) {
            cb_push_back(cb_s, 1);
            cb_push_back(cb_l, 1);
        } else {
            cb_push_back(cb_p, 1);
        }
        for (uint32_t t = 0; t < seg;) {
            const uint32_t nb0 = BATCH - pushed % BATCH, nb = nb0 < seg - t ? nb0 : seg - t;
            cb_reserve_back(cb_x, nb);
            const uint32_t l1 = get_write_ptr(cb_x);
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(xs.get_noc_addr((tt0 + t + i) * Xw + ct), l1 + i * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, nb);
            // each tile but the segment's last is also the next tile's previous one: a local copy into c_6
            for (uint32_t i = 0; i < nb; ++i) {
                if (t + i + 1 < seg) {
                    cb_reserve_back(cb_p, 1);
                    noc_async_write(l1 + i * 2048, me | get_write_ptr(cb_p), 2048);
                    noc_async_write_barrier();
                    cb_push_back(cb_p, 1);
                }
            }
            t += nb;
            pushed += nb;
        }
        u += seg;
    }
}

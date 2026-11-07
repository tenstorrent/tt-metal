// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv (4 taps), reader (NCRISC). A core owns whole channel tile-columns ct and walks
// their time tiles in order: per column the conv state tile (rows 0-2 = the 3 previous tokens) and the 4 tap row
// tiles, then the input tiles (tt, ct) of x (the in_proj output, its first Dt tile columns are q | k | v).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: x, state, taps, consts, Dt (channel tile columns), ncores, grid_x, Tt, Xw (x width in tiles)
    const uint32_t Dt = get_common_arg_val<uint32_t>(4);
    uint32_t first, n;
    core_span(Dt, get_common_arg_val<uint32_t>(5), get_common_arg_val<uint32_t>(6), 1, first, n);
    const uint32_t Tt = get_common_arg_val<uint32_t>(7), Xw = get_common_arg_val<uint32_t>(8);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ws = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), 2048);
    constexpr uint32_t cb_x = 0, cb_s = 1, cb_w = 2, cb_c = 3, NCONST = 11;

    cb_reserve_back(cb_c, NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc_async_read(cs.get_noc_addr(i), get_write_ptr(cb_c) + i * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, NCONST);
    for (uint32_t ct = first; ct < first + n; ++ct) {
        cb_reserve_back(cb_s, 1);
        cb_reserve_back(cb_w, 4);
        noc_async_read(ss.get_noc_addr(ct), get_write_ptr(cb_s), 2048);
        for (uint32_t k = 0; k < 4; ++k) {
            noc_async_read(ws.get_noc_addr(k * Dt + ct), get_write_ptr(cb_w) + k * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
        cb_push_back(cb_w, 4);
        for (uint32_t tt = 0; tt < Tt; ++tt) {
            cb_reserve_back(cb_x, 1);
            noc_async_read(xs.get_noc_addr(tt * Xw + ct), get_write_ptr(cb_x), 2048);
            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
        }
    }
}

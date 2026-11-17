// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert 5 x 5 groups (ws_g5.py), weight injector reader (BRISC): loads the P weight sets' slices of its target
// core index j into CB 0 once ([p][WTOT_MAX] tiles): gate/up = W_gu[K tiles K0 .. K0 + KSL, tile columns GU0 ..
// GU0 + NW] ([KSL][NW], gate / up interleaved), then down = W_d[KTD][down columns DS .. DS + DC] ([KTD][DC]).
// RT: 0 wgu addr, 1 wd addr, 2 K0, 3 KSL, 4 GU0, 5 DS, 6 DC. CT: 0 KT, 1 KTD, 2 HT, 3 NW, 4 W_gu tile columns, 5 P,
// 6 WTOT_MAX, then TensorAccessorArgs(wgu), (wd)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KTD = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2), NW = get_compile_time_arg_val(3);
    constexpr uint32_t GUC = get_compile_time_arg_val(4), P = get_compile_time_arg_val(5);
    constexpr uint32_t WTOT_MAX = get_compile_time_arg_val(6);
    constexpr auto gargs = TensorAccessorArgs<7>();
    constexpr auto dargs = TensorAccessorArgs<gargs.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(0);
    const auto gs = TensorAccessor(gargs, get_arg_val<uint32_t>(0), tb);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(1), tb);
    const uint32_t K0 = get_arg_val<uint32_t>(2), KSL = get_arg_val<uint32_t>(3), GU0 = get_arg_val<uint32_t>(4);
    const uint32_t DS = get_arg_val<uint32_t>(5), DC = get_arg_val<uint32_t>(6);
    cb_reserve_back(0, P * WTOT_MAX);
    const uint32_t pre = get_write_ptr(0);
    for (uint32_t p = 0; p < P; ++p) {
        uint32_t l1 = pre + p * WTOT_MAX * tb;
        for (uint32_t k = 0; k < KSL; ++k) {
            for (uint32_t n = 0; n < NW; ++n, l1 += tb) {
                noc_async_read_page((p * KT + K0 + k) * GUC + GU0 + n, gs, l1);
            }
        }
        for (uint32_t k = 0; k < KTD; ++k) {
            for (uint32_t n = 0; n < DC; ++n, l1 += tb) {
                noc_async_read_page((p * KTD + k) * HT + DS + n, ds, l1);
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(0, P * WTOT_MAX);
}

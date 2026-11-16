// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert shared expert, cluster core BRISC (NOC0: DRAM reads only). Preloads this core's resident weights: its
// NPAIR gate/up pairs (KT x GW tiles of the interleaved gate/up matrix, CB 1) and its DC down columns (KTD x DC tiles
// from column C0, CB 3).
// RT: 0 w addr, 1 wd addr, 2 J, 3 C0. CT: 0 KT, 1 GW, 2 WCOLS (gate/up matrix tile columns), 3 KTD, 4 DC, 5 HT, then
// TensorAccessorArgs(w), (wd).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr uint32_t WCOLS = get_compile_time_arg_val(2);
    constexpr uint32_t KTD = get_compile_time_arg_val(3);
    constexpr uint32_t DC = get_compile_time_arg_val(4);
    constexpr uint32_t HT = get_compile_time_arg_val(5);
    constexpr auto wargs = TensorAccessorArgs<6>();
    constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t w_cb = tt::CBIndex::c_1, wd_cb = tt::CBIndex::c_3;
    const uint32_t J = get_arg_val<uint32_t>(2), C0 = get_arg_val<uint32_t>(3);
    const uint32_t wb = get_tile_size(w_cb), db = get_tile_size(wd_cb);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(0), wb);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(1), db);
    cb_reserve_back(w_cb, KT * GW);
    uint32_t l1 = get_write_ptr(w_cb);
    for (uint32_t k = 0; k < KT; ++k) {
        for (uint32_t t = 0; t < GW; ++t, l1 += wb) {
            noc_async_read_page(k * WCOLS + J * GW + t, ws, l1);
        }
    }
    cb_reserve_back(wd_cb, KTD * DC);
    l1 = get_write_ptr(wd_cb);
    for (uint32_t k = 0; k < KTD; ++k) {
        for (uint32_t t = 0; t < DC; ++t, l1 += db) {
            noc_async_read_page(k * HT + C0 + t, ds, l1);
        }
    }
    noc_async_read_barrier();
    cb_push_back(w_cb, KT * GW);
    cb_push_back(wd_cb, KTD * DC);
}

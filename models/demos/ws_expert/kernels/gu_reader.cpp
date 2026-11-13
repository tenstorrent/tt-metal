// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert gate/up reader (NCRISC, NOC1). Bench preload: this grid row's x store (XT tiles) into CB 0 and the pair
// group's gate/up weights [KT][GW] into CB 1, both resident. Then, if this core's last piece starts from the next
// core's partial (PRE), per expert: reserve a slot of the partial ring (CB 5, 2 slots), grant it to the next core,
// wait for the data (ARRIVED), push it to compute.
// RT: 0 x addr, 1 w addr, 2 x store index (pages store * XT + t), 3 has_pre, 4 next x, 5 next y (NOC coords), 6 J
// CT: 0 KT, 1 GW, 2 XT, 3 WCOLS (tiles a W row), 4 (unused), 5 NEXP, then TensorAccessorArgs(x), TensorAccessorArgs(w)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t pin_cb = tt::CBIndex::c_5;
constexpr uint32_t SEM_GRANT = 1, SEM_ARR = 2;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr uint32_t XT = get_compile_time_arg_val(2);
    constexpr uint32_t WCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t NEXP = get_compile_time_arg_val(5);
    constexpr auto xargs = TensorAccessorArgs<6>();
    constexpr auto wargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    const uint32_t xaddr = get_arg_val<uint32_t>(0);
    const uint32_t waddr = get_arg_val<uint32_t>(1);
    const uint32_t store = get_arg_val<uint32_t>(2);
    const uint32_t has_pre = get_arg_val<uint32_t>(3);
    const uint32_t nx = get_arg_val<uint32_t>(4), ny = get_arg_val<uint32_t>(5);
    const uint32_t J = get_arg_val<uint32_t>(6);

    const uint32_t xbytes = get_tile_size(x_cb), wbytes = get_tile_size(w_cb);
    const auto xs = TensorAccessor(xargs, xaddr, xbytes);
    const auto ws = TensorAccessor(wargs, waddr, wbytes);
    cb_reserve_back(x_cb, XT);
    uint32_t l1 = get_write_ptr(x_cb);
    for (uint32_t t = 0; t < XT; ++t) {
        noc_async_read_page(store * XT + t, xs, l1);
        l1 += xbytes;
    }
    cb_reserve_back(w_cb, KT * GW);
    l1 = get_write_ptr(w_cb);
    for (uint32_t k = 0; k < KT; ++k) {
        for (uint32_t t = 0; t < GW; ++t) {
            noc_async_read_page(k * WCOLS + J * GW + t, ws, l1);
            l1 += wbytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(x_cb, XT);
    cb_push_back(w_cb, KT * GW);

    if (has_pre) {
        volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARR));
        const uint64_t next_grant = get_noc_addr(nx, ny, get_semaphore(SEM_GRANT));
        for (uint32_t e = 0; e < NEXP; ++e) {
            cb_reserve_back(pin_cb, GW);
            noc_semaphore_inc(next_grant, 1);
            noc_semaphore_wait_min(arr, e + 1);
            cb_push_back(pin_cb, GW);
        }
        noc_async_atomic_barrier();
    }
}

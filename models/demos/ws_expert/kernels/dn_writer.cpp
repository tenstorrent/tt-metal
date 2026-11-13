// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert down writer (NCRISC, NOC1): pops the CW output tiles of every row; the last expert's go to DRAM
// (out [R rows][NOUT] tiles; the h ring's place p holds row order[p]).
// RT: 0 out addr, 1 column tile base, 2.. order[R]. CT: 0 CW, 1 NOUT, 2 R, 3 NEXP, then TensorAccessorArgs(out)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t out_cb = tt::CBIndex::c_16;

void kernel_main() {
    constexpr uint32_t CW = get_compile_time_arg_val(0);
    constexpr uint32_t NOUT = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t NEXP = get_compile_time_arg_val(3);
    constexpr auto oargs = TensorAccessorArgs<4>();
    const uint32_t oaddr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t obytes = get_tile_size(out_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);
    for (uint32_t e = 0; e < NEXP; ++e) {
        for (uint32_t p = 0; p < R; ++p) {
            cb_wait_front(out_cb, CW);
            if (e + 1 == NEXP) {
                const uint32_t row = get_arg_val<uint32_t>(2 + p);
                uint32_t l1 = get_read_ptr(out_cb);
                for (uint32_t t = 0; t < CW; ++t) {
                    noc_async_write_page(row * NOUT + col0 + t, os, l1);
                    l1 += obytes;
                }
                noc_async_write_barrier();
            }
            cb_pop_front(out_cb, CW);
        }
    }
}

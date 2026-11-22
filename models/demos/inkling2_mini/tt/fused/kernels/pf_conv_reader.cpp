// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill causal k=4 conv over T rows, reader. Unit = (column tile j, tile rows R0..R1). Per unit: the 4 tap tiles
// of column j (fp32, [1, C] tap rows: row 0 valid); per row R: x[R, j] (bf16), the row above it (x[R - 1, j], or the
// 32-row tail for R = 0) and, in resid / hres mode, resid[R, j] (res_page bytes: fp32 / bf16). Plus once: the 6
// constant shift tiles (bf16). Runtime args: x, tail, taps (4 x [32, C] fp32), shifts, resid (0 = none), n_units, (j,
// R0, R1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t has_resid = get_named_compile_time_arg_val("has_resid");
    constexpr uint32_t RP = get_named_compile_time_arg_val("res_page");
    constexpr uint32_t cb_x = 0, cb_prev = 1, cb_taps = 2, cb_sh = 3, cb_res = 4;
    const uint32_t x = get_arg_val<uint32_t>(0), tail = get_arg_val<uint32_t>(1);
    const uint32_t shifts = get_arg_val<uint32_t>(6), resid = get_arg_val<uint32_t>(7);
    uint32_t taps[4];
    for (uint32_t r = 0; r < 4; ++r) {
        taps[r] = get_arg_val<uint32_t>(2 + r);
    }
    const uint32_t n_units = get_arg_val<uint32_t>(8);
    const InterleavedAddrGen<true> xa = {.bank_base_address = x, .page_size = 2048};
    const InterleavedAddrGen<true> ta = {.bank_base_address = tail, .page_size = 2048};
    const InterleavedAddrGen<true> sa = {.bank_base_address = shifts, .page_size = 2048};
    const InterleavedAddrGen<true> ra = {.bank_base_address = resid, .page_size = RP};

    cb_reserve_back(cb_sh, 6);
    for (uint32_t i = 0; i < 6; ++i) {
        noc_async_read(sa.get_noc_addr(i), get_write_ptr(cb_sh) + i * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(cb_sh, 6);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t j = get_arg_val<uint32_t>(9 + 3 * u), r0 = get_arg_val<uint32_t>(10 + 3 * u);
        const uint32_t r1 = get_arg_val<uint32_t>(11 + 3 * u);
        cb_reserve_back(cb_taps, 4);
        for (uint32_t r = 0; r < 4; ++r) {
            const InterleavedAddrGen<true> tp = {.bank_base_address = taps[r], .page_size = 4096};  // [1, C] rows
            noc_async_read(tp.get_noc_addr(j), get_write_ptr(cb_taps) + r * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_taps, 4);
        for (uint32_t R = r0; R < r1; ++R) {
            cb_reserve_back(cb_x, 1);
            cb_reserve_back(cb_prev, 1);
            noc_async_read(xa.get_noc_addr(R * WT + j), get_write_ptr(cb_x), 2048);
            if (R == 0) {
                noc_async_read(ta.get_noc_addr(j), get_write_ptr(cb_prev), 2048);
            } else {
                noc_async_read(xa.get_noc_addr((R - 1) * WT + j), get_write_ptr(cb_prev), 2048);
            }
            if constexpr (has_resid) {
                cb_reserve_back(cb_res, 1);
                noc_async_read(ra.get_noc_addr(R * WT + j), get_write_ptr(cb_res), RP);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
            cb_push_back(cb_prev, 1);
            if constexpr (has_resid) {
                cb_push_back(cb_res, 1);
            }
        }
    }
}

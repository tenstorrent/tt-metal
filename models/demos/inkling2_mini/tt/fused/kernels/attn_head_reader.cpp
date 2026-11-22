// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, head role, reader. Per item (one head): nlag rows of 128 projection columns -> dense bf16
// pages (cb_x), their taps / scale (dense fp32 pages, cb_t), and with `dyn` the per-token taps of global layers as
// dense fp32 pages filled with d_l (cb_d). Plus the reduce scaler and (has_tau) the tau tile.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t nlag = get_named_compile_time_arg_val("nlag");
    constexpr uint32_t dyn = get_named_compile_time_arg_val("dyn");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t cb_x = 0, cb_t = 1, cb_d = 2, cb_scaler = 3, cb_tau = 4, cb_stage = 24;
    constexpr uint32_t n = nitems * nlag;

    uint32_t a = 0;
    const uint32_t proj = get_arg_val<uint32_t>(a++);
    const uint32_t taps = get_arg_val<uint32_t>(a++);
    const uint32_t scaler = get_arg_val<uint32_t>(a++);
    const uint32_t tau = get_arg_val<uint32_t>(a++);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    cb_reserve_back(cb_scaler, 1);
    attnio::read_pages(scaler, 4096, 0, 1, get_write_ptr(cb_scaler));
    if constexpr (has_tau) {
        cb_reserve_back(cb_tau, 1);
        attnio::read_pages(tau, 4096, 0, 1, get_write_ptr(cb_tau));
    }
    cb_reserve_back(cb_t, n);
    cb_reserve_back(cb_x, n);
    const uint32_t xw = get_write_ptr(cb_x);
    const uint32_t tw = get_write_ptr(cb_t);
    attnio::zero_l1(xw, n * 2048);
    for (uint32_t i = 0; i < nitems; ++i) {
        const uint32_t t0 = get_arg_val<uint32_t>(a++);    // projection tile of the head's first column
        const uint32_t r0 = get_arg_val<uint32_t>(a++);    // window row of lag 0
        const uint32_t page = get_arg_val<uint32_t>(a++);  // first taps page
        attnio::read_pages(taps, 4096, page, nlag, tw + i * nlag * 4096);
        noc_async_read_barrier();  // zeros in place before the gather lands
        for (uint32_t l = 0; l < nlag; ++l) {
            attnio::gather_row(proj, 2048, 2, t0, 4, r0 + l, xw + (i * nlag + l) * 2048, stage);
        }
        if constexpr (dyn) {
            const uint32_t dcol = get_arg_val<uint32_t>(a++);  // column of d_0 (lags follow) in row r0
            cb_reserve_back(cb_d, nlag);
            const uint32_t dw = get_write_ptr(cb_d);
            attnio::zero_l1(dw, nlag * 4096);
            noc_async_read_barrier();
            const uint32_t e = attnio::read_elem_block(proj, 2048, 2, r0, dcol, stage);
            volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(e);
            for (uint32_t l = 0; l < nlag; ++l) {
                const uint32_t bits = static_cast<uint32_t>(d16[l]) << 16;  // bf16 -> fp32
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dw + l * 4096);
                for (uint32_t k = 0; k < 128; ++k) {
                    p[k] = bits;
                }
            }
            cb_push_back(cb_d, nlag);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_scaler, 1);
    if constexpr (has_tau) {
        cb_push_back(cb_tau, 1);
    }
    cb_push_back(cb_t, n);
    cb_push_back(cb_x, n);
}

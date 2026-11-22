// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, head role, reader. Per item (one head): nlag rows of 128 projection columns -> dense bf16
// pages (cb_x), their taps / scale (dense fp32 pages, cb_t), and with `dyn` the per-token taps of global layers as
// dense fp32 pages filled with d_l (cb_d). Plus the reduce scaler, (has_tau) the tau tile and (mass) the
// far-cache mass page.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t nlag = get_named_compile_time_arg_val("nlag");
    constexpr uint32_t dyn = get_named_compile_time_arg_val("dyn");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t mass = get_named_compile_time_arg_val("mass");  // far-cache mass chunk page (cb_mass)
    constexpr uint32_t cb_x = 0, cb_t = 1, cb_d = 2, cb_scaler = 3, cb_tau = 4, cb_mass = 5, cb_stage = 24;
    constexpr uint32_t n = nitems * nlag;

    uint32_t a = 0;
    const uint32_t proj = get_arg_val<uint32_t>(a++);
    const uint32_t taps = get_arg_val<uint32_t>(a++);
    const uint32_t scaler = get_arg_val<uint32_t>(a++);
    const uint32_t tau = get_arg_val<uint32_t>(a++);
    const uint32_t mass_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    cb_reserve_back(cb_scaler, 1);
    attnio::read_pages(scaler, 4096, 0, 1, get_write_ptr(cb_scaler));
    if constexpr (has_tau) {
        cb_reserve_back(cb_tau, 1);
        attnio::read_pages(tau, 4096, 0, 1, get_write_ptr(cb_tau));
    }
    if constexpr (mass) {
        cb_reserve_back(cb_mass, 1);
        attnio::read_pages(mass_addr, 4096, 0, 1, get_write_ptr(cb_mass));
    }
    cb_reserve_back(cb_t, n);
    cb_reserve_back(cb_x, n);
    const uint32_t xw = get_write_ptr(cb_x);
    const uint32_t tw = get_write_ptr(cb_t);
    // wave 1: zeros, taps, the staged row blocks and (dyn) the d block - one read barrier
    attnio::zero_l1(xw, n * 2048);
    uint32_t t0s[nitems], r0s[nitems], dcol = 0, d_elem = 0, dw = 0;
    for (uint32_t i = 0; i < nitems; ++i) {
        t0s[i] = get_arg_val<uint32_t>(a++);               // projection tile of the head's first column
        r0s[i] = get_arg_val<uint32_t>(a++);               // window row of lag 0
        const uint32_t page = get_arg_val<uint32_t>(a++);  // first taps page
        attnio::read_pages(taps, 4096, page, nlag, tw + i * nlag * 4096);
        for (uint32_t l = 0; l < nlag; ++l) {
            attnio::stage_row(proj, 2048, 2, t0s[i], 4, r0s[i] + l, stage + (i * nlag + l) * 512);
        }
        if constexpr (dyn) {                    // nitems == 1
            dcol = get_arg_val<uint32_t>(a++);  // column of d_0 (lags follow) in row r0
            cb_reserve_back(cb_d, nlag);
            dw = get_write_ptr(cb_d);
            attnio::zero_l1(dw, nlag * 4096);
            d_elem = attnio::stage_elem_block(proj, 2048, 2, r0s[i], dcol, stage + n * 512);
        }
    }
    noc_async_read_barrier();
    // wave 2: chunks into the zeroed pages, d pages filled
    for (uint32_t i = 0; i < nitems; ++i) {
        for (uint32_t l = 0; l < nlag; ++l) {
            const uint32_t k = i * nlag + l;
            attnio::unstage_row(2, 4, r0s[i] + l, xw + k * 2048, stage + k * 512);
        }
    }
    if constexpr (dyn) {
        volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d_elem);
        for (uint32_t l = 0; l < nlag; ++l) {
            const uint32_t bits = static_cast<uint32_t>(d16[l]) << 16;  // bf16 -> fp32
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dw + l * 4096);
            for (uint32_t k = 0; k < 128; ++k) {
                p[k] = bits;
            }
        }
        cb_push_back(cb_d, nlag);
    }
    noc_async_read_barrier();
    cb_push_back(cb_scaler, 1);
    if constexpr (has_tau) {
        cb_push_back(cb_tau, 1);
    }
    if constexpr (mass) {
        cb_push_back(cb_mass, 1);
    }
    cb_push_back(cb_t, n);
    cb_push_back(cb_x, n);
}

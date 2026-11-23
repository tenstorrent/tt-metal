// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused block attention residual, reader: partial row -> dense, reduce scaler, queries, sealed stack rows 0..k-1.
#include "rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");  // 0: seal only
    constexpr uint32_t k = get_named_compile_time_arg_val("k");    // row of the partial = number of sealed rows read
    constexpr uint32_t has_weight = get_named_compile_time_arg_val("has_weight");  // rms weight (dense fp32)
    constexpr uint32_t cb_p = 0, cb_q = 1, cb_s = 2, cb_scaler = 3, cb_wt = 4, cb_stage = 24;

    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t p_tile = get_arg_val<uint32_t>(1);
    const uint32_t p_es = get_arg_val<uint32_t>(2);
    const uint32_t p_row = get_arg_val<uint32_t>(3);    // row-form: tile row; dense twin: first page
    const uint32_t p_dense = get_arg_val<uint32_t>(4);  // 1: p is a dense twin (fused-kernel output)
    const uint32_t q_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(7);
    const uint32_t w_addr = get_arg_val<uint32_t>(8);

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), rowio::DRAM_RD_ALIGN);

    {
        DeviceZoneScopedN("R_gather");
        cb_reserve_back(cb_scaler, 1);
        rowio::read_pages(scaler_addr, 4096, 0, 1, get_write_ptr(cb_scaler));
        cb_reserve_back(cb_p, T);
        if (p_dense) {
            rowio::read_pages(p_addr, p_tile, p_row, T, get_write_ptr(cb_p));
            noc_async_read_barrier();
        } else {
#ifdef ATTNRES_SPLIT_GATHER  // gatez: the writer gathers tiles ntc/2.. and pushes CB 29 when they have landed
            rowio::gather_row(
                p_addr, p_tile, p_es, ntc, p_row, get_write_ptr(cb_p), T * 1024 * p_es, stage, 0, ntc / 2);
            cb_wait_front(29, 1);
            cb_pop_front(29, 1);
#else
            rowio::gather_row(p_addr, p_tile, p_es, ntc, p_row, get_write_ptr(cb_p), T * 1024 * p_es, stage);
#endif
        }
        cb_push_back(cb_scaler, 1);
        cb_push_back(cb_p, T);
    }
    if constexpr (has_weight) {
        cb_reserve_back(cb_wt, T);
        rowio::read_pages(w_addr, 4096, 0, T, get_write_ptr(cb_wt));
        noc_async_read_barrier();
        cb_push_back(cb_wt, T);
    }

    if constexpr (nq > 0) {
        DeviceZoneScopedN("R_qs");
        cb_reserve_back(cb_q, nq * T);
        rowio::read_pages(q_addr, 4096, 0, nq * T, get_write_ptr(cb_q));
        if constexpr (k > 0) {
            cb_reserve_back(cb_s, k * T);
            rowio::read_pages(s_addr, 4096, 0, k * T, get_write_ptr(cb_s));
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, nq * T);
        if constexpr (k > 0) {
            cb_push_back(cb_s, k * T);
        }
    }

    if constexpr (get_named_compile_time_arg_val("split_out")) {
        // row norm (E10d): first half of the output row scatter (the writer does the rest)
        const uint32_t out_addr = get_arg_val<uint32_t>(9), out_tile = get_arg_val<uint32_t>(10);
        const uint32_t out_es = get_arg_val<uint32_t>(11), out_row = get_arg_val<uint32_t>(12);
        cb_wait_front(16, T);
        constexpr bool dram = get_named_compile_time_arg_val("split_out") != 2;  // 2: L1-interleaved output
        rowio::scatter_row<dram>(out_addr, out_tile, out_es, ntc, out_row, get_read_ptr(16), 0, ntc / 2);
        noc_async_write_barrier();
    }

    if constexpr (get_named_compile_time_arg_val("has_norm")) {
        // MoE input norm (args 9..13): its weight (dense fp32), then the first half of the bf16 L1 output row
        const uint32_t nw_addr = get_arg_val<uint32_t>(9), n_addr = get_arg_val<uint32_t>(10);
        const uint32_t n_tile = get_arg_val<uint32_t>(11), n_es = get_arg_val<uint32_t>(12);
        const uint32_t n_row = get_arg_val<uint32_t>(13);  // tile row of the output (verify row r: r)
        constexpr uint32_t cb_nw = 5, cb_nn = 19;
        cb_reserve_back(cb_nw, T);
        rowio::read_pages(nw_addr, 4096, 0, T, get_write_ptr(cb_nw));
        noc_async_read_barrier();
        cb_push_back(cb_nw, T);
        cb_wait_front(cb_nn, T);
        rowio::scatter_row<false>(n_addr, n_tile, n_es, ntc, n_row, get_read_ptr(cb_nn), 0, ntc / 2);
        noc_async_write_barrier();
    }
}

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
            rowio::gather_row(p_addr, p_tile, p_es, ntc, p_row, get_write_ptr(cb_p), T * 1024 * p_es, stage);
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
}

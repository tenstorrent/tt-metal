// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused short conv, reader: x row -> dense (first half of a row-form x; the writer gathers the rest and reads
// the residual), taps and the 3 previous ring slots.
#include "rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");  // C / 32
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");  // dense pages per vector
    constexpr uint32_t cb_x = 0, cb_ring = 1, cb_taps = 2, cb_sync = 25, cb_stage = 24;

    // x only; the writer gathers the second half of a row-form x and reads ring, taps and resid
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_tile = get_arg_val<uint32_t>(1);
    const uint32_t x_es = get_arg_val<uint32_t>(2);
    const uint32_t x_row = get_arg_val<uint32_t>(3);
    const uint32_t x_dense = get_arg_val<uint32_t>(4);
    const uint32_t r1 = get_arg_val<uint32_t>(5);  // ring slots at lag 1, 2, 3
    const uint32_t r2 = get_arg_val<uint32_t>(6);
    const uint32_t r3 = get_arg_val<uint32_t>(7);
    const uint32_t taps = get_arg_val<uint32_t>(8);

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), rowio::DRAM_RD_ALIGN);

    DeviceZoneScopedN("R_x");
    cb_reserve_back(cb_x, T);
    if (x_dense == 1) {
        rowio::read_pages(x_addr, x_tile, x_row, T, get_write_ptr(cb_x));
        noc_async_read_barrier();
    } else {
        if (x_dense == 2) {  // row form in L1
            rowio::gather_row<false>(
                x_addr, x_tile, x_es, ntc, x_row, get_write_ptr(cb_x), T * 1024 * x_es, stage, 0, ntc / 2);
        } else {
            rowio::gather_row(
                x_addr, x_tile, x_es, ntc, x_row, get_write_ptr(cb_x), T * 1024 * x_es, stage, 0, ntc / 2);
        }
    }
    cb_reserve_back(cb_taps, 4 * T);
    rowio::read_pages(taps, 4096, 0, 4 * T, get_write_ptr(cb_taps));
    // own > 0 (verify row r of an R-row step): lags 1..own are the x of rows r-1.. of this step (args 12.., 5 per
    // row, as x's), not ring slots; the compute takes them from cb_px exactly as the ring would hold them
    constexpr uint32_t own = get_named_compile_time_arg_val("own");
    constexpr uint32_t cb_px = 4;
    cb_reserve_back(cb_ring, 3 * T);
    const uint32_t ring = get_write_ptr(cb_ring);
    if constexpr (own < 1) {
        rowio::read_pages(r1, 4096, 0, T, ring);
    }
    if constexpr (own < 2) {
        rowio::read_pages(r2, 4096, 0, T, ring + T * 4096);
    }
    if constexpr (own < 3) {
        rowio::read_pages(r3, 4096, 0, T, ring + 2 * T * 4096);
    }
    if constexpr (own > 0) {
        cb_reserve_back(cb_px, own * T);
        for (uint32_t k = 0; k < own; ++k) {
            const uint32_t a = 12 + 5 * k;
            const uint32_t p_addr = get_arg_val<uint32_t>(a), p_tile = get_arg_val<uint32_t>(a + 1);
            const uint32_t p_es = get_arg_val<uint32_t>(a + 2), p_row = get_arg_val<uint32_t>(a + 3);
            const uint32_t p_kind = get_arg_val<uint32_t>(a + 4);
            const uint32_t dst = get_write_ptr(cb_px) + k * T * 1024 * p_es;
            if (p_kind == 1) {
                rowio::read_pages(p_addr, p_tile, p_row, T, dst);
            } else if (p_kind == 2) {
                rowio::gather_row<false>(p_addr, p_tile, p_es, ntc, p_row, dst, T * 1024 * p_es, stage);
            } else {
                rowio::gather_row(p_addr, p_tile, p_es, ntc, p_row, dst, T * 1024 * p_es, stage);
            }
        }
    }
    noc_async_read_barrier();
    cb_wait_front(cb_sync, 1);  // the writer's half has landed
    cb_pop_front(cb_sync, 1);
    cb_push_back(cb_x, T);
    cb_push_back(cb_taps, 4 * T);
    cb_push_back(cb_ring, 3 * T);
    if constexpr (own > 0) {
        cb_push_back(cb_px, own * T);
    }

    // first half of the output row scatter (the writer does the rest)
    constexpr uint32_t row_out = get_named_compile_time_arg_val("row_out");
    if constexpr (row_out) {
        constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
        const uint32_t out_addr = get_arg_val<uint32_t>(9), out_tile = get_arg_val<uint32_t>(10);
        const uint32_t out_es = get_arg_val<uint32_t>(11);
        cb_wait_front(cb_out, T);
        rowio::scatter_row(out_addr, out_tile, out_es, ntc, 0, get_read_ptr(cb_out), 0, ntc / 2);
        noc_async_write_barrier();
    }
}

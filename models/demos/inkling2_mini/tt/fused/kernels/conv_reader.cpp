// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused short conv, reader: x row -> dense, taps + 3 previous ring slots (dense pages), optional residual row.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");  // C / 32
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");  // dense pages per vector
    constexpr uint32_t has_resid = get_named_compile_time_arg_val("has_resid");
    constexpr uint32_t cb_x = 0, cb_ring = 1, cb_taps = 2, cb_resid = 3, cb_stage = 24;

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t x_tile = get_arg_val<uint32_t>(a++);
    const uint32_t x_es = get_arg_val<uint32_t>(a++);
    const uint32_t r1 = get_arg_val<uint32_t>(a++);  // ring slots at lag 1, 2, 3
    const uint32_t r2 = get_arg_val<uint32_t>(a++);
    const uint32_t r3 = get_arg_val<uint32_t>(a++);
    const uint32_t taps = get_arg_val<uint32_t>(a++);
    const uint32_t res_addr = get_arg_val<uint32_t>(a++);
    const uint32_t res_tile = get_arg_val<uint32_t>(a++);
    const uint32_t res_es = get_arg_val<uint32_t>(a++);

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), rowio::DRAM_RD_ALIGN);

    cb_reserve_back(cb_x, T);
    rowio::gather_row(x_addr, x_tile, x_es, ntc, 0, get_write_ptr(cb_x), T * 1024 * x_es, stage);
    cb_push_back(cb_x, T);

    cb_reserve_back(cb_taps, 4 * T);
    rowio::read_pages(taps, 4096, 0, 4 * T, get_write_ptr(cb_taps));
    cb_reserve_back(cb_ring, 3 * T);
    const uint32_t ring = get_write_ptr(cb_ring);
    rowio::read_pages(r1, 4096, 0, T, ring);
    rowio::read_pages(r2, 4096, 0, T, ring + T * 4096);
    rowio::read_pages(r3, 4096, 0, T, ring + 2 * T * 4096);
    noc_async_read_barrier();
    cb_push_back(cb_taps, 4 * T);
    cb_push_back(cb_ring, 3 * T);

    if constexpr (has_resid) {
        cb_reserve_back(cb_resid, T);
        rowio::gather_row(res_addr, res_tile, res_es, ntc, 0, get_write_ptr(cb_resid), T * 1024 * res_es, stage);
        cb_push_back(cb_resid, T);
    }
}

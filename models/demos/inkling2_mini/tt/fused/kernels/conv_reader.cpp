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

    // x only; the writer gathers the second half of a row-form x and reads ring, taps and resid (E10c)
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
    if (x_dense) {
        rowio::read_pages(x_addr, x_tile, x_row, T, get_write_ptr(cb_x));
        noc_async_read_barrier();
    } else {
        rowio::gather_row(x_addr, x_tile, x_es, ntc, x_row, get_write_ptr(cb_x), T * 1024 * x_es, stage, 0, ntc / 2);
    }
    cb_reserve_back(cb_taps, 4 * T);
    rowio::read_pages(taps, 4096, 0, 4 * T, get_write_ptr(cb_taps));
    cb_reserve_back(cb_ring, 3 * T);
    const uint32_t ring = get_write_ptr(cb_ring);
    rowio::read_pages(r1, 4096, 0, T, ring);
    rowio::read_pages(r2, 4096, 0, T, ring + T * 4096);
    rowio::read_pages(r3, 4096, 0, T, ring + 2 * T * 4096);
    noc_async_read_barrier();
    cb_wait_front(cb_sync, 1);  // the writer's half has landed
    cb_pop_front(cb_sync, 1);
    cb_push_back(cb_x, T);
    cb_push_back(cb_taps, 4 * T);
    cb_push_back(cb_ring, 3 * T);
}

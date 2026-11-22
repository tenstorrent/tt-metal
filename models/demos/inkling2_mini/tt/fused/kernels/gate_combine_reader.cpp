// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate combine, reader (one core per nt column tiles): z (+ block mask), the core's forget / gain column
// slices, and its h / m row-form tiles.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");    // z width in tiles (64 * gates / 32)
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");    // column tiles per core
    constexpr uint32_t nct = get_named_compile_time_arg_val("nct");  // C / 32
    constexpr uint32_t has_mask = get_named_compile_time_arg_val("has_mask");
    constexpr uint32_t h_tile = get_named_compile_time_arg_val("h_tile");  // page bytes of h and m
    constexpr uint32_t cb_z = 0, cb_mask = 1, cb_f = 2, cb_g = 3, cb_h = 4, cb_m = 5;

    const uint32_t z_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t f_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t h_addr = get_arg_val<uint32_t>(4);
    const uint32_t m_addr = get_arg_val<uint32_t>(5);
    const uint32_t col0 = get_arg_val<uint32_t>(6);

    cb_reserve_back(cb_z, kz);
    rowio::read_pages(z_addr, 4096, 0, kz, get_write_ptr(cb_z));
    if constexpr (has_mask) {
        cb_reserve_back(cb_mask, kz);
        rowio::read_pages(mask_addr, 4096, 0, kz, get_write_ptr(cb_mask));
    }
    noc_async_read_barrier();
    cb_push_back(cb_z, kz);
    if constexpr (has_mask) {
        cb_push_back(cb_mask, kz);
    }

    cb_reserve_back(cb_f, kz * nt);
    cb_reserve_back(cb_g, kz * nt);
    const uint32_t f = get_write_ptr(cb_f), g = get_write_ptr(cb_g);
    for (uint32_t kt = 0; kt < kz; ++kt) {
        rowio::read_pages(f_addr, 4096, kt * nct + col0, nt, f + kt * nt * 4096);
        rowio::read_pages(g_addr, 4096, kt * nct + col0, nt, g + kt * nt * 4096);
    }
    cb_reserve_back(cb_h, nt);
    cb_reserve_back(cb_m, nt);
    rowio::read_pages(h_addr, h_tile, col0, nt, get_write_ptr(cb_h));
    rowio::read_pages(m_addr, h_tile, col0, nt, get_write_ptr(cb_m));
    noc_async_read_barrier();
    cb_push_back(cb_f, kz * nt);
    cb_push_back(cb_g, kz * nt);
    cb_push_back(cb_h, nt);
    cb_push_back(cb_m, nt);
}

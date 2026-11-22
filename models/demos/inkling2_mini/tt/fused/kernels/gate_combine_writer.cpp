// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate combine, writer: rows 0..R-1 of the core's output tiles -> row-form fp32 DRAM tensor (the
// padding rows of the zeroed output buffer are never written).
#include "rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_out = 17;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t dense_addr = get_arg_val<uint32_t>(2);  // dense twin (E10c): vector r = pages r*T .., fp32
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");

    const InterleavedAddrGen<true> dst = {.bank_base_address = out_addr, .page_size = 4096};
    const InterleavedAddrGen<true> dd = {.bank_base_address = dense_addr, .page_size = 4096};
    {  // G for the compute (the reader reads F, z, h, m)
        DeviceZoneScopedN("W_g");
        constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
        constexpr uint32_t nct = get_named_compile_time_arg_val("nct");
        constexpr uint32_t cb_g = 3;
        constexpr uint32_t fg = get_named_compile_time_arg_val("fg_page");  // page bytes of F and G
        const uint32_t g_addr = get_arg_val<uint32_t>(3);
        constexpr uint32_t f_w = get_named_compile_time_arg_val("f_w");
        if constexpr (get_named_compile_time_arg_val("fused")) {  // F / G after z's weights (set by the z root)
            noc_semaphore_wait(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(get_named_compile_time_arg_val("sem_go_w"))),
                1);
        }
        if constexpr (f_w > 0) {  // F rows kt < f_w into the reader's (not yet pushed) cb_f pages
            const uint32_t f_addr = get_arg_val<uint32_t>(4);
            const uint32_t f = get_write_ptr(2);
            cb_reserve_back(6, 1);
            for (uint32_t kt = 0; kt < f_w; ++kt) {
                rowio::read_pages(f_addr, fg, kt * nct + col0, nt, f + kt * nt * fg);
            }
        }
        cb_reserve_back(cb_g, kz * nt);
        const uint32_t g = get_write_ptr(cb_g);
        for (uint32_t kt = 0; kt < kz; ++kt) {
            rowio::read_pages(g_addr, fg, kt * nct + col0, nt, g + kt * nt * fg);
        }
        noc_async_read_barrier();
        if constexpr (f_w > 0) {
            cb_push_back(6, 1);
        }
        cb_push_back(cb_g, kz * nt);
    }
    {
        DeviceZoneScopedN("W_wait");
        cb_wait_front(cb_out, nt);
    }
    DeviceZoneScopedN("W_out");
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t j = 0; j < nt; ++j) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t off = rowio::row_offset(r, 4);
            noc_async_write(src + j * 4096 + off, dst.get_noc_addr(col0 + j, off), 64);
            noc_async_write(src + j * 4096 + off + 1024, dst.get_noc_addr(col0 + j, off + 1024), 64);
            const uint32_t e = 32 * (col0 + j);  // first element of this tile column in vector r
            const uint64_t d = dd.get_noc_addr(r * T + e / 1024, (e % 1024) * 4);
            noc_async_write(src + j * 4096 + off, d, 64);
            noc_async_write(src + j * 4096 + off + 1024, d + 64, 64);
        }
    }
    noc_async_write_barrier();
}

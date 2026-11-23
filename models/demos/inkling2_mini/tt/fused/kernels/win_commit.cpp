// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Multi-token verify commit of the conv window (one core per window shard): rows 4 (n - 1) .. 4 (n - 1) + 3 of the
// pass's window tiles (scratch, see attn_window_rows.cpp) -> rows 0..3 of the window state's tiles, which is the
// state plain decode leaves after n steps of the single-input window (rows 4..31 untouched).
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");
    const uint32_t hist = get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t buf = get_write_ptr(0), src = get_write_ptr(1);
    const InterleavedAddrGen<true> h = {.bank_base_address = hist, .page_size = 2048};
    const InterleavedAddrGen<true> sc = {.bank_base_address = scratch, .page_size = 2048};
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(h.get_noc_addr(t0 + j), buf + j * 2048, 2048);
        noc_async_read(sc.get_noc_addr(t0 + j), src + j * 2048, 2048);
    }
    noc_async_read_barrier();
    const uint32_t r0 = 4 * (n - 1);
    for (uint32_t j = 0; j < ntiles; ++j) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(get_noc_addr(src + j * 2048 + f * 512 + r0 * 32), buf + j * 2048 + f * 512, 128);
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_write(buf + j * 2048, h.get_noc_addr(t0 + j), 2048);
    }
    noc_async_write_barrier();
}

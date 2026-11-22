// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Global-layer near softmax, writer: the dense bf16 p row -> row r of head h's p_near tiles ([1, 8, 3, 1024] bf16
// TILE, DRAM), and lse -> row r, column 0 of head h's lse tile ([1, 8, 3, 1] fp32 TILE, DRAM; columns 1..3 get 0).
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");
    constexpr uint32_t cb_p = 28, cb_lse = 29, cb_scr = 6;
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t lse_addr = get_arg_val<uint32_t>(1);
    const uint32_t h = get_arg_val<uint32_t>(2);
    const uint32_t r = get_arg_val<uint32_t>(3);

    const InterleavedAddrGen<true> pa = {.bank_base_address = p_addr, .page_size = 2048};
    const InterleavedAddrGen<true> la = {.bank_base_address = lse_addr, .page_size = 4096};
    cb_wait_front(cb_p, 1);
    const uint32_t src = get_read_ptr(cb_p);
    for (uint32_t j = 0; j < nt; ++j) {  // bf16 row r (< 16): 16 values in face 0, 16 in face 1 (+512 B)
        noc_async_write(src + j * 64, pa.get_noc_addr(h * nt + j, r * 32), 32);
        noc_async_write(src + j * 64 + 32, pa.get_noc_addr(h * nt + j, 512 + r * 32), 32);
    }
    cb_wait_front(cb_lse, 1);
    cb_reserve_back(cb_scr, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scr));
    s[0] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_lse));
    s[1] = s[2] = s[3] = 0;
    noc_async_write(get_write_ptr(cb_scr), la.get_noc_addr(h, r * 64), 16);
    noc_async_write_barrier();
    cb_pop_front(cb_p, 1);
    cb_pop_front(cb_lse, 1);
}

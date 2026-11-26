// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Global-layer near softmax, reader (one core per (kv head h, q row r)): row r of head h's near scores
// ([1, 8, 3, 1024] fp32 TILE, DRAM) -> dense page A; the mask row 3h + r ([1, 1, 24, 1024] bf16 TILE, DRAM) ->
// dense bf16 page; page B = -inf with the sink block's row r ([1, 8, 3, 32] fp32) in its first 32 slots;
// plus the constant ones tile (fp32) and reduce scaler (bf16).
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");  // score tiles per row (32)
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_bias = 2, cb_ones = 3, cb_scaler = 4, cb_stage = 5;
    const uint32_t near = get_arg_val<uint32_t>(0);
    const uint32_t mask = get_arg_val<uint32_t>(1);
    const uint32_t sink = get_arg_val<uint32_t>(2);
    const uint32_t consts = get_arg_val<uint32_t>(3);  // dense fp32 pages: [ones, -inf]
    const uint32_t scaler = get_arg_val<uint32_t>(4);  // bf16 ones tile
    const uint32_t h = get_arg_val<uint32_t>(5);
    const uint32_t r = get_arg_val<uint32_t>(6);
    const uint32_t mrow = get_arg_val<uint32_t>(7);  // 3h + r

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    cb_reserve_back(cb_bias, 1);
    cb_reserve_back(cb_ones, 1);
    cb_reserve_back(cb_scaler, 1);
    const InterleavedAddrGen<true> nr = {.bank_base_address = near, .page_size = 4096};
    const InterleavedAddrGen<true> sk = {.bank_base_address = sink, .page_size = 4096};
    const InterleavedAddrGen<true> cs = {.bank_base_address = consts, .page_size = 4096};
    const InterleavedAddrGen<true> sc = {.bank_base_address = scaler, .page_size = 2048};
    const uint32_t a = get_write_ptr(cb_a), b = get_write_ptr(cb_b);
    noc_async_read(cs.get_noc_addr(0), get_write_ptr(cb_ones), 4096);
    noc_async_read(cs.get_noc_addr(1), b, 4096);
    noc_async_read(sc.get_noc_addr(0), get_write_ptr(cb_scaler), 2048);
    for (uint32_t j = 0; j < nt; ++j) {  // fp32 row r (< 16): 16 values in face 0, 16 in face 1 (+1024 B)
        noc_async_read(nr.get_noc_addr(h * nt + j, r * 64), a + j * 128, 64);
        noc_async_read(nr.get_noc_addr(h * nt + j, 1024 + r * 64), a + j * 128 + 64, 64);
    }
    noc_async_read_barrier();  // page B's -inf must land before the sink row goes on top of it
    noc_async_read(sk.get_noc_addr(h, r * 64), b, 64);
    noc_async_read(sk.get_noc_addr(h, 1024 + r * 64), b + 64, 64);
    rowio::gather_row(mask, 2048, 2, nt, mrow, get_write_ptr(cb_bias), 2048, get_write_ptr(cb_stage));
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_bias, 1);
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_scaler, 1);
}

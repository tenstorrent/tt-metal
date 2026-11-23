// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Speculative-verify copy of attn_window.cpp: the same window update, but the old window is read from `hist` and the
// new one written to `hist_out` (arg 6), so each verify token leaves its own window and the state is only replaced
// by the commit of the accepted tokens. Everything else is attn_window.cpp unchanged:
//
// Fused attention conv window (data movement only, one core per projection-input shard). Per window tile
// (32 rows x 32 columns of the [1, 1, 32, C] bf16 window): the old tile with each lag group of rows shifted down by
// one (rows r0 .. r0 + 2 -> r0 + 1 .. r0 + 3, both 16-column faces) and the step's input row written into row r0
// (and the q input into row 0 in triple mode). The tile goes back to the window state (DRAM) and into this core's
// L1 shard of the projection input, so the projection needs no shift matmul, row adds or interleaved-to-sharded.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");    // window tiles of this core
    constexpr uint32_t ngroups = get_named_compile_time_arg_val("ngroups");  // lag groups: 1 (single), 2 (triple)
    constexpr uint32_t cb_old = 0, cb_new = 1, cb_stage = 24;

    uint32_t a = 0;
    const uint32_t hist = get_arg_val<uint32_t>(a++);
    const uint32_t shard = get_arg_val<uint32_t>(a++);
    const uint32_t t0 = get_arg_val<uint32_t>(a++);  // first window tile of this core
    const uint32_t xq = get_arg_val<uint32_t>(a++);  // input rows (row form, bf16); triple: q, then k, v
    const uint32_t xk = get_arg_val<uint32_t>(a++);
    const uint32_t xv = get_arg_val<uint32_t>(a++);
    const uint32_t hist_out = get_arg_val<uint32_t>(a++);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    const InterleavedAddrGen<true> h = {.bank_base_address = hist, .page_size = 2048};
    const InterleavedAddrGen<true> ho = {.bank_base_address = hist_out, .page_size = 2048};
    const InterleavedAddrGen<true> q = {.bank_base_address = xq, .page_size = 2048};
    const InterleavedAddrGen<true> k = {.bank_base_address = xk, .page_size = 2048};
    const InterleavedAddrGen<true> v = {.bank_base_address = xv, .page_size = 2048};
    const uint32_t old = get_write_ptr(cb_old), nw = get_write_ptr(cb_new);
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(h.get_noc_addr(t0 + j), old + j * 2048, 2048);
        for (uint32_t f = 0; f < 2; ++f) {  // input row chunks: 64B blocks at face f row 0
            const uint32_t s = stage + (j * 2 + f) * 192;
            noc_async_read(q.get_noc_addr(t0 + j, f * 512), s, 64);
            if constexpr (ngroups == 2) {
                noc_async_read(k.get_noc_addr(t0 + j, f * 512), s + 64, 64);
                noc_async_read(v.get_noc_addr(t0 + j, f * 512), s + 128, 64);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(get_noc_addr(old + j * 2048), nw + j * 2048, 2048);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        const uint32_t o = old + j * 2048, n = nw + j * 2048;
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t s = stage + (j * 2 + f) * 192;
            const uint32_t fo = f * 512;
            if constexpr (ngroups == 1) {  // rows 0..3: one lag group at row 0
                noc_async_read(get_noc_addr(o + fo), n + fo + 32, 96);
                noc_async_read(get_noc_addr(s), n + fo, 32);
            } else {  // row 0 = q; k lags at rows 1..4, v lags at rows 5..8
                noc_async_read(get_noc_addr(s), n + fo, 32);
                noc_async_read(get_noc_addr(o + fo + 32), n + fo + 64, 96);
                noc_async_read(get_noc_addr(s + 64), n + fo + 32, 32);
                noc_async_read(get_noc_addr(o + fo + 160), n + fo + 192, 96);
                noc_async_read(get_noc_addr(s + 128), n + fo + 160, 32);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_write(nw + j * 2048, ho.get_noc_addr(t0 + j), 2048);
        noc_async_write(nw + j * 2048, get_noc_addr(shard + j * 2048), 2048);
    }
    noc_async_write_barrier();
}

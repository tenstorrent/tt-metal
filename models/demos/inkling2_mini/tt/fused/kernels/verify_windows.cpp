// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Speculative verify: the conv windows of R tokens in one program, stacked into the projection input.
// Per core (its `ntiles` window tiles, as attn_window.cpp): token t's window = token t-1's window with each lag group
// shifted down one row and token t's input row(s) written in (attn_window.cpp's update, same bytes); it is written
// to hist_out[t] (DRAM, the state the commit of t+1 accepted tokens installs) and its rows 0..8 go to rows
// 9t..9t+8 of this core's shard of the projection input (L1), so one projection serves all R tokens.
// Token 0 starts from `hist` (the state). Rows >= 9R of the shard are not written (the projection's output rows
// are independent; only rows < 9R are read back).
// Runtime args: hist, shard, t0, then per token: xq, xk, xv, hist_out.
#include "attn_io.hpp"

namespace {
constexpr uint32_t TILE = 2048;

// Copy window row r (both 16-column halves) of tile `src` to row rr of tile `dst` (local L1).
FORCE_INLINE void copy_row(uint32_t src, uint32_t r, uint32_t dst, uint32_t rr) {
    const uint32_t so = (r >> 4) * 1024 + (r & 15) * 32, doff = (rr >> 4) * 1024 + (rr & 15) * 32;
    noc_async_read(get_noc_addr(src + so), dst + doff, 32);
    noc_async_read(get_noc_addr(src + so + 512), dst + doff + 512, 32);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");
    constexpr uint32_t ngroups = get_named_compile_time_arg_val("ngroups");  // 1 (single), 2 (triple)
    constexpr uint32_t R = get_named_compile_time_arg_val("rows");
    constexpr uint32_t win_rows = 9;
    constexpr uint32_t cb_old = 0, cb_new = 1, cb_stage = 24;

    const uint32_t hist = get_arg_val<uint32_t>(0);
    const uint32_t shard = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    const InterleavedAddrGen<true> h = {.bank_base_address = hist, .page_size = TILE};
    uint32_t old = get_write_ptr(cb_old), nw = get_write_ptr(cb_new);
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(h.get_noc_addr(t0 + j), old + j * TILE, TILE);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < R; ++t) {
        const uint32_t a = 3 + 4 * t;
        const InterleavedAddrGen<true> q = {.bank_base_address = get_arg_val<uint32_t>(a), .page_size = TILE};
        const InterleavedAddrGen<true> k = {.bank_base_address = get_arg_val<uint32_t>(a + 1), .page_size = TILE};
        const InterleavedAddrGen<true> v = {.bank_base_address = get_arg_val<uint32_t>(a + 2), .page_size = TILE};
        const InterleavedAddrGen<true> ho = {.bank_base_address = get_arg_val<uint32_t>(a + 3), .page_size = TILE};
        for (uint32_t j = 0; j < ntiles; ++j) {
            for (uint32_t f = 0; f < 2; ++f) {  // input row chunks: 64B blocks at face f row 0
                const uint32_t s = stage + (j * 2 + f) * 192;
                noc_async_read(q.get_noc_addr(t0 + j, f * 512), s, 64);
                if constexpr (ngroups == 2) {
                    noc_async_read(k.get_noc_addr(t0 + j, f * 512), s + 64, 64);
                    noc_async_read(v.get_noc_addr(t0 + j, f * 512), s + 128, 64);
                }
            }
            noc_async_read(get_noc_addr(old + j * TILE), nw + j * TILE, TILE);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < ntiles; ++j) {  // attn_window.cpp's update of the copy
            const uint32_t o = old + j * TILE, n = nw + j * TILE;
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t s = stage + (j * 2 + f) * 192;
                const uint32_t fo = f * 512;
                if constexpr (ngroups == 1) {
                    noc_async_read(get_noc_addr(o + fo), n + fo + 32, 96);
                    noc_async_read(get_noc_addr(s), n + fo, 32);
                } else {
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
            noc_async_write(nw + j * TILE, ho.get_noc_addr(t0 + j), TILE);
            for (uint32_t r = 0; r < win_rows; ++r) {
                copy_row(nw + j * TILE, r, shard + j * TILE, win_rows * t + r);
            }
        }
        noc_async_write_barrier();
        noc_async_read_barrier();
        const uint32_t tmp = old;  // this token's window is the next token's old window
        old = nw;
        nw = tmp;
    }
}

// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Conv window for a multi-token verify pass (data movement only, one core per projection-input shard). Token t's
// rows start at row b = G t of each window tile, G = 4 (single) or 9 (triple), laid out as attn_window.cpp lays out
// rows 0 .. G - 1 at that token's step:
// - single: row b + l = the input of token t - l (the pass's input xs[t - l], or the window state's row l - t - 1
//   for t - l < 0);
// - triple: row b = q input of token t; k lag l at row b + 1 + l, v lag l at b + 5 + l (inputs of token t - l, or
//   the window state's k row l - t / v row 4 + l - t for t - l < 0).
// The rest of the tile is zero. The tile goes to this core's L1 shard
// of the projection input and to `scratch` (DRAM, for the commit); the window state is not written.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");    // window tiles of this core
    constexpr uint32_t R = get_named_compile_time_arg_val("R");              // tokens, R * G <= 32
    constexpr uint32_t ngroups = get_named_compile_time_arg_val("ngroups");  // lag groups: 1 (single), 2 (triple)
    constexpr uint32_t nin = ngroups == 1 ? 1 : 3;                           // input rows per token
    constexpr uint32_t G = ngroups == 1 ? 4 : 9;                             // window rows per token
    static_assert(R * G <= 32);
    constexpr uint32_t cb_old = 0, cb_new = 1, cb_stage = 24;

    const uint32_t hist = get_arg_val<uint32_t>(0);
    const uint32_t shard = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);  // first window tile of this core
    const uint32_t scratch = get_arg_val<uint32_t>(3);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const InterleavedAddrGen<true> h = {.bank_base_address = hist, .page_size = 2048};
    const InterleavedAddrGen<true> sc = {.bank_base_address = scratch, .page_size = 2048};
    const uint32_t old = get_write_ptr(cb_old), nw = get_write_ptr(cb_new);
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_read(h.get_noc_addr(t0 + j), old + j * 2048, 2048);
        for (uint32_t t = 0; t < R * nin; ++t) {  // input t = token t / nin, input t % nin (q, k, v)
            const InterleavedAddrGen<true> x = {.bank_base_address = get_arg_val<uint32_t>(4 + t), .page_size = 2048};
            for (uint32_t f = 0; f < 2; ++f) {  // input row: 64B block at face f row 0
                noc_async_read(x.get_noc_addr(t0 + j, f * 512), stage + ((j * R * nin + t) * 2 + f) * 64, 64);
            }
        }
        attnio::zero_l1(nw + j * 2048, 2048);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        const uint32_t o = old + j * 2048, n = nw + j * 2048;
        // one window row: input `in` of token `tok` (staged), or window state row `srow`
        // (rows 16..31 live in faces 2 / 3; srow < 16)
        auto put = [&](uint32_t row, bool staged, uint32_t tok, uint32_t in, uint32_t srow) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t src =
                    staged ? stage + ((j * R * nin + tok * nin + in) * 2 + f) * 64 : o + f * 512 + srow * 32;
                noc_async_read(get_noc_addr(src), n + (row / 16) * 1024 + f * 512 + (row % 16) * 32, 32);
            }
        };
        for (uint32_t t = 0; t < R; ++t) {
            const uint32_t b = G * t;
            if constexpr (ngroups == 1) {
                for (uint32_t l = 0; l < 4; ++l) {
                    put(b + l, t >= l, t - l, 0, l - t - 1);
                }
            } else {
                put(b, true, t, 0, 0);
                for (uint32_t l = 0; l < 4; ++l) {
                    put(b + 1 + l, t >= l, t - l, 1, l - t);
                    put(b + 5 + l, t >= l, t - l, 2, 4 + l - t);
                }
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ntiles; ++j) {
        noc_async_write(nw + j * 2048, get_noc_addr(shard + j * 2048), 2048);
        noc_async_write(nw + j * 2048, sc.get_noc_addr(t0 + j), 2048);
    }
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Speculative verify commit of every attention app in one program (tt/verify.py commit_attn): a list of jobs per
// core, each the work of one core of kv_commit.cpp (ring rows of kv group g) or of win_commit.cpp (one window
// shard), with the same reads, row copies and writes.
// Runtime args: pos0 address, n (accepted tokens), number of kv jobs, number of window jobs, then per kv job
// (pending, k ring, v ring, g) and per window job (hist, scratch, first tile, G).
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t tb = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr uint32_t width = get_named_compile_time_arg_val("width");
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");
    constexpr uint32_t S = seq_tiles * 32;
    const uint32_t pos_addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t nkv = get_arg_val<uint32_t>(2);
    const uint32_t nwin = get_arg_val<uint32_t>(3);
    const uint32_t stage = align_power_of_2(get_write_ptr(24), attnio::ALIGN);

    uint32_t a = 4;
    if (nkv) {
        const uint32_t pend_l1 = stage, tile_l1 = stage + 2 * width * tb, pos_l1 = tile_l1 + 2 * tb;
        const InterleavedAddrGen<true> pp = {.bank_base_address = pos_addr, .page_size = 4};
        noc_async_read(pp.get_noc_addr(0, 0), pos_l1, 16);
        noc_async_read_barrier();
        const uint32_t pos0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
        for (uint32_t j = 0; j < nkv; ++j, a += 4) {
            const InterleavedAddrGen<true> pa = {.bank_base_address = get_arg_val<uint32_t>(a), .page_size = tb};
            const uint32_t ring[2] = {get_arg_val<uint32_t>(a + 1), get_arg_val<uint32_t>(a + 2)};
            const uint32_t g = get_arg_val<uint32_t>(a + 3);
            for (uint32_t kv = 0; kv < 2; ++kv) {
                for (uint32_t c = 0; c < width; ++c) {
                    noc_async_read(pa.get_noc_addr((kv * 8 + g) * width + c), pend_l1 + (kv * width + c) * tb, tb);
                }
            }
            noc_async_read_barrier();
            for (uint32_t t = 0; t < n; ++t) {
                const uint32_t slot = (pos0 + t) % S;
                for (uint32_t kv = 0; kv < 2; ++kv) {
                    const InterleavedAddrGen<true> ra = {.bank_base_address = ring[kv], .page_size = tb};
                    for (uint32_t c = 0; c < width; ++c) {
                        const uint32_t page = (g * seq_tiles + (slot >> 5)) * width + c;
                        noc_async_read(ra.get_noc_addr(page), tile_l1, tb);
                        noc_async_read_barrier();
                        attnio::copy_tile_row(pend_l1 + (kv * width + c) * tb, t, tile_l1, slot & 31, tb);
                        noc_async_write(tile_l1, ra.get_noc_addr(page), tb);
                        noc_async_write_barrier();
                    }
                }
            }
        }
    }
    const uint32_t buf = stage, src = stage + ntiles * 2048;
    for (uint32_t j = 0; j < nwin; ++j, a += 4) {
        const InterleavedAddrGen<true> h = {.bank_base_address = get_arg_val<uint32_t>(a), .page_size = 2048};
        const InterleavedAddrGen<true> sc = {.bank_base_address = get_arg_val<uint32_t>(a + 1), .page_size = 2048};
        const uint32_t t0 = get_arg_val<uint32_t>(a + 2);
        const uint32_t G = get_arg_val<uint32_t>(a + 3);
        for (uint32_t i = 0; i < ntiles; ++i) {
            noc_async_read(h.get_noc_addr(t0 + i), buf + i * 2048, 2048);
            noc_async_read(sc.get_noc_addr(t0 + i), src + i * 2048, 2048);
        }
        noc_async_read_barrier();
        const uint32_t r0 = G * (n - 1);  // rows r0 .. r0 + G - 1 -> rows 0 .. G - 1 (win_commit.cpp)
        for (uint32_t i = 0; i < ntiles; ++i) {
            for (uint32_t l = 0; l < G; ++l) {
                const uint32_t s = r0 + l;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t so = (s / 16) * 1024 + f * 512 + (s % 16) * 32;
                    const uint32_t dof = (l / 16) * 1024 + f * 512 + (l % 16) * 32;
                    noc_async_read(get_noc_addr(src + i * 2048 + so), buf + i * 2048 + dof, 32);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < ntiles; ++i) {
            noc_async_write(buf + i * 2048, h.get_noc_addr(t0 + i), 2048);
        }
        noc_async_write_barrier();
    }
}

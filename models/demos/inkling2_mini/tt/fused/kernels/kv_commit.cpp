// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Multi-token verify commit (kv group g = arg 4): rows 0 .. n - 1 of the pending K / V tiles (pending [2, 8, 32, 128]:
// batch g = K, batch 8 + g = V, row t = token t) -> row (pos0 + t) % S of the ring tiles [1, 8, S, 128] (TILE, DRAM).
// Read-modify-write of whole tiles: bf16 face-row words, or bfp8 exponent byte + 16 mantissa bytes per face row.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t n = get_named_compile_time_arg_val("n");
    constexpr uint32_t tb = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr uint32_t width = get_named_compile_time_arg_val("width");
    const uint32_t pend = get_arg_val<uint32_t>(0);
    const uint32_t ring[2] = {get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t g = get_arg_val<uint32_t>(4);
    const uint32_t stage = align_power_of_2(get_write_ptr(24), attnio::ALIGN);
    const uint32_t pend_l1 = stage, tile_l1 = stage + 2 * width * tb, pos_l1 = tile_l1 + 2 * tb;

    const InterleavedAddrGen<true> pa = {.bank_base_address = pend, .page_size = tb};
    for (uint32_t kv = 0; kv < 2; ++kv) {
        for (uint32_t c = 0; c < width; ++c) {
            noc_async_read(pa.get_noc_addr((kv * 8 + g) * width + c), pend_l1 + (kv * width + c) * tb, tb);
        }
    }
    const InterleavedAddrGen<true> pp = {.bank_base_address = pos_addr, .page_size = 4};
    noc_async_read(pp.get_noc_addr(0, 0), pos_l1, 16);
    noc_async_read_barrier();
    const uint32_t pos0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    constexpr uint32_t S = seq_tiles * 32;
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

// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Drafter block K/V writer: rows 0..n-1 of head h of src [1, H, 32, 128] (TILE bf16, interleaved DRAM) -> rows
// pos .. pos + n - 1 of head h of dst [1, H, S, 128] (TILE bf16, interleaved DRAM), pos read on device from an
// int32 tensor (trace safe). One core per (tensor, head); pure data movement, 32 B face-row pieces.
// ring > 0: dst is a ring held twice, row x goes to rows x % ring and x % ring + ring.
// flat: src is [1, 1, H, 128], one position with head h as its row h (n = 1).
// Tile layout: row r, chunk c (16 elements) of a 32x32 bf16 tile sits at byte ((r / 16) * 2 + c) * 512 + (r % 16) * 32.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TILE_BYTES = 2048, W = 4;  // 128 = 4 tiles per head row
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t seq_tiles = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t h = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    const uint32_t ring = get_arg_val<uint32_t>(6);
    const uint32_t flat = get_arg_val<uint32_t>(7);
    const uint32_t stage = get_write_ptr(0);  // 64 B pos + n * W * 2 pieces of 64 B

    const InterleavedAddrGen<true> pa = {.bank_base_address = pos_addr, .page_size = 4};
    noc_async_read(pa.get_noc_addr(0, 0), stage, 64);
    const InterleavedAddrGen<true> s = {.bank_base_address = src, .page_size = TILE_BYTES};
    const InterleavedAddrGen<true> d = {.bank_base_address = dst, .page_size = TILE_BYTES};
    uint32_t buf = stage + 64;
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t sr = flat ? h : r;  // the source row
        for (uint32_t t = 0; t < W; ++t) {
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t off = (((sr >> 4) * 2 + c) * 256 + (sr & 15) * 16) * 2;
                // DRAM reads keep the source's offset modulo 64 in L1
                noc_async_read(s.get_noc_addr((flat ? 0 : h * W) + t, off), buf + (off & 63), 32);
                buf += 64;
            }
        }
    }
    noc_async_read_barrier();
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    buf = stage + 64;
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t sr = flat ? h : r;
        const uint32_t row0 = ring ? (pos + r) % ring : pos + r;
        for (uint32_t t = 0; t < W; ++t) {
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t off = (((sr >> 4) * 2 + c) * 256 + (sr & 15) * 16) * 2;
                for (uint32_t row = row0; row <= row0 + ring; row += ring ? ring : 1) {
                    const uint32_t tr = row >> 5, rr = row & 31;
                    const uint32_t doff = (((rr >> 4) * 2 + c) * 256 + (rr & 15) * 16) * 2;
                    noc_async_write(buf + (off & 63), d.get_noc_addr((h * seq_tiles + tr) * W + t, doff), 32);
                }
                buf += 64;
            }
        }
    }
    noc_async_write_barrier();
}

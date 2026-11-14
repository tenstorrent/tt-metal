// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched DFlash context K/V ring write (Laguna; ring_write.ring_write_batched): row r of each new [1, nkv, T, hd] K or V
// tensor goes to ring slot dests[r] % RING of user dests[r] / RING in its [U, nkv, RING, hd] ring (dests[r] =
// 0xFFFFFFFF: row r is not written). Core c handles ring c / (nkv * hdt), head (c / hdt) % nkv, column tile c % hdt:
// it reads the source tiles of its column and its row group, then walks the rows, reading each destination ring tile
// when a row first lands in it and writing it back when the next row lands elsewhere. Row r of a 32x32 bf16 tile is 32 bytes at face (r / 16) * 2, offset (r % 16) * 32, and
// 32 more in the next face (512 bytes later). Common runtime args: dests address, T, then the R source addresses, then
// the R ring addresses. src_batched: the sources are [P, nkv, 32, hd] (row r is row r % 32 of batch r / 32) instead of
// [1, nkv, T, hd].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t nkv = get_compile_time_arg_val(2);
    constexpr uint32_t hdt = get_compile_time_arg_val(3);       // head_dim tiles
    constexpr uint32_t ring_len = get_compile_time_arg_val(4);  // RING (slots per user)
    constexpr uint32_t max_t = get_compile_time_arg_val(5);     // source rows (a multiple of 32)
    constexpr uint32_t src_batched = get_compile_time_arg_val(6);
    // rows are split over row groups of group_rows rows (a multiple of the rows per user, so no two groups write
    // one ring tile): core c handles group c / (R * nkv * hdt)
    constexpr uint32_t group_rows = get_compile_time_arg_val(7);
    constexpr uint32_t ring_rows = ring_len / 32;
    constexpr uint32_t src_rows = max_t / 32;
    constexpr auto i_args = TensorAccessorArgs<8>();
    constexpr auto s_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    constexpr uint32_t units = R * nkv * hdt;
    constexpr uint32_t groups = (max_t + group_rows - 1) / group_rows;
    if (c >= units * groups) {
        return;
    }
    const uint32_t grp = c / units, cu = c % units;
    const uint32_t ring = cu / (nkv * hdt), h = (cu / hdt) % nkv, j = cu % hdt;
    const uint32_t T = get_common_arg_val<uint32_t>(1);
    const uint32_t r_begin = grp * group_rows;
    const uint32_t r_end = r_begin + group_rows < T ? r_begin + group_rows : T;
    const auto idx = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), max_t * 4);
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(2 + ring), 2048);
    const auto dst = TensorAccessor(r_args, get_common_arg_val<uint32_t>(2 + R + ring), 2048);
    // cb 0: this group's source tiles then one ring tile; cb 1: the T destination words
    const uint32_t sbuf = get_write_ptr(0), ibuf = get_write_ptr(1);
    const uint32_t t0 = r_begin / 32, t1 = (r_end + 31) / 32;
    const uint32_t dbuf = sbuf + (t1 - t0) * 2048;
    noc_async_read(idx.get_noc_addr(0), ibuf, max_t * 4);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t st = src_batched ? (t * nkv + h) * hdt + j : (h * src_rows + t) * hdt + j;
        noc_async_read(src.get_noc_addr(st), sbuf + (t - t0) * 2048, 2048);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* dests = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ibuf);
    // one pass over the rows: a ring tile is read when the first row bound for it comes up and written back when a
    // row bound for another tile (or the end) comes up; a user's rows are consecutive, so a tile is usually read once
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t r = r_begin; r < r_end; ++r) {
        const uint32_t dr = dests[r];
        if (dr == 0xFFFFFFFFu) {
            continue;
        }
        const uint32_t user = dr / ring_len, tr = (dr % ring_len) / 32;
        const uint32_t tile = ((user * nkv + h) * ring_rows + tr) * hdt + j;
        if (tile != cur) {
            if (cur != 0xFFFFFFFFu) {
                noc_async_write(dbuf, dst.get_noc_addr(cur), 2048);
                noc_async_write_barrier();
            }
            noc_async_read(dst.get_noc_addr(tile), dbuf, 2048);
            noc_async_read_barrier();
            cur = tile;
        }
        const uint32_t d = dr % 32, rr = r % 32;
        const uint32_t so = (r / 32 - t0) * 2048 + (rr / 16) * 1024 + (rr % 16) * 32;
        const uint32_t dofs = (d / 16) * 1024 + (d % 16) * 32;
        for (uint32_t half = 0; half < 2; ++half) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbuf + so + half * 512);
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbuf + dofs + half * 512);
            for (uint32_t w = 0; w < 8; ++w) {
                o[w] = s[w];
            }
        }
    }
    if (cur != 0xFFFFFFFFu) {
        noc_async_write(dbuf, dst.get_noc_addr(cur), 2048);
        noc_async_write_barrier();
    }
}

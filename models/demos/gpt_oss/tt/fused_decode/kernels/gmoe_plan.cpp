// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE plan (dflash/grouped_moe.py GroupedMoE.plan): one core per slot s (BRISC) writes slot s's stripes of the
// gather and combine matrices (BF16 tiles):
//   P1T [MAX_SLOTS * 32, ROWS]: row 32 s + i = one-hot of the source row of slot row i (zero for padding rows);
//                               X_g = P1T x gathers every slot's rows (tiles s * 8 + c, c = 0 .. 7);
//   Pw  [ROWS, MAX_SLOTS * 32]: column 32 s + i = the routing weight w[src][e] at row src; out = Pw Y_g sums each row's
//                               routed experts' outputs (tiles t * MAX_SLOTS + s, t = 0 .. 7).
// The routed experts of a row are exactly the dense router's (TopKRouter.dense_decode: every expert whose logit is >=
// the k-th largest, so a BF16 tie at the k-th logit routes k + 1 experts): the router's top-(k + 1) ids, the last one
// only when its dense weight is nonzero; the weights are the dense ones, bit for bit. Expert e's routed rows (row
// order) fill tiles[e] = ceil(count / 32) slots from base[e] (expert order); slot s of expert e holds its rows ranked
// 32 (s - base[e]) .. + 31. Core 0 also writes the plan table [tiles[E], base[E]] (the grouped matmuls' readers).
//
// common runtime args: [ids, w, p1t, pw, table]
// compile-time args: [cb_scr, GX, MAX_SLOTS, accessors (ids, w, p1t, pw, table)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

void kernel_main() {
    const uint32_t ids_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p1t_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t pw_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t GX = get_compile_time_arg_val(1);
    constexpr uint32_t MAX_SLOTS = get_compile_time_arg_val(2);
    constexpr auto ids_args = TensorAccessorArgs<3>();
    constexpr auto w_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto p1t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto pw_args = TensorAccessorArgs<p1t_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<pw_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    constexpr uint32_t RT = gmoe::ROWS / 32;  // row tiles
    constexpr uint32_t E = gmoe::E;
    constexpr uint32_t K = gmoe::K_IDS;  // ids per row (top-k + 1)

    const uint32_t s = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const auto s_ids = TensorAccessor(ids_args, ids_addr, tile);
    const auto s_w = TensorAccessor(w_args, w_addr, tile);
    const auto s_p1t = TensorAccessor(p1t_args, p1t_addr, tile);
    const auto s_pw = TensorAccessor(pw_args, pw_addr, tile);
    const auto s_table = TensorAccessor(table_args, table_addr, 8 * E);

    cb_reserve_back(cb_scr, 1);
    const uint32_t ids_l1 = get_write_ptr(cb_scr);
    const uint32_t w_l1 = ids_l1 + RT * tile;
    const uint32_t p1t_l1 = w_l1 + RT * tile;   // stripe: RT tiles [32 slot rows, 32 rows]
    const uint32_t pw_l1 = p1t_l1 + RT * tile;  // stripe: RT tiles [32 rows, 32 slot rows]
    const uint32_t rid_l1 = pw_l1 + RT * tile;  // per row: K id bytes (0xFF: an unrouted (k + 1)-th), 8 bytes
    const uint32_t table_l1 = rid_l1 + 8 * gmoe::ROWS;
    for (uint32_t t = 0; t < RT; ++t) {
        noc_async_read(s_ids.get_noc_addr(t), ids_l1 + t * tile, tile);
        noc_async_read(s_w.get_noc_addr(t), w_l1 + t * tile, tile);
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < 2 * RT * tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, p1t_l1 + z, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    // byte offset of element (row, col) in a BF16 / UINT16 tile (faces of 16 x 16)
    auto off = [](uint32_t row, uint32_t col) -> uint32_t {
        return ((row / 16) * 2 + col / 16) * 512 + ((row % 16) * 16 + col % 16) * 2;
    };
    auto u16 = [](uint32_t a) -> volatile tt_l1_ptr uint16_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a);
    };
    // the routed ids per row and the expert counts
    uint8_t* rid = reinterpret_cast<uint8_t*>(rid_l1);
    uint32_t cnt[E];
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t r = 0; r < gmoe::ROWS; ++r) {
        const uint32_t* w32 = reinterpret_cast<const uint32_t*>(ids_l1 + (r / 32) * tile + off(r % 32, 0));
        const uint32_t a = w32[0], b = w32[1], c = w32[2];  // ids (0, 1), (2, 3), (4, 5)
        const uint32_t id[K] = {a & 0xFF, (a >> 16) & 0xFF, b & 0xFF, (b >> 16) & 0xFF, c & 0xFF};
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t e = id[k] & (E - 1);
            if (k == K - 1 && (*u16(w_l1 + (r / 32) * tile + off(r % 32, e)) & 0x7FFF) == 0) {
                e = 0xFF;  // the (k + 1)-th expert, not routed (no tie)
            } else {
                cnt[e]++;
            }
            rid[8 * r + k] = static_cast<uint8_t>(e);
        }
    }
    gmoe::Plan p;
    uint32_t base = 0;
    for (uint32_t e = 0; e < E; ++e) {
        p.tiles[e] = (cnt[e] + 31) / 32;
        p.base[e] = base;
        base += p.tiles[e];
    }
    if (s == 0) {  // the plan table: [tiles[E], base[E]]
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
        for (uint32_t e = 0; e < E; ++e) {
            t[e] = p.tiles[e];
            t[E + e] = p.base[e];
        }
        noc_async_write(table_l1, s_table.get_noc_addr(0), 8 * E);
    }

    uint32_t e = E;
    for (uint32_t x = 0; x < E; ++x) {
        if (p.tiles[x] && p.base[x] <= s && s < p.base[x] + p.tiles[x]) {
            e = x;
        }
    }
    if (e < E) {
        const uint32_t lo = 32 * (s - p.base[e]), hi = lo + 32;
        uint32_t rank = 0;
        for (uint32_t r = 0; r < gmoe::ROWS && rank < hi; ++r) {
            for (uint32_t k = 0; k < K; ++k) {
                if (rid[8 * r + k] != e) {
                    continue;
                }
                if (rank >= lo) {
                    const uint32_t i = rank - lo;
                    *u16(p1t_l1 + (r / 32) * tile + off(i, r % 32)) = 0x3F80;  // 1.0
                    *u16(pw_l1 + (r / 32) * tile + off(r % 32, i)) = *u16(w_l1 + (r / 32) * tile + off(r % 32, e));
                }
                ++rank;
            }
        }
    }
    for (uint32_t t = 0; t < RT; ++t) {
        noc_async_write(p1t_l1 + t * tile, s_p1t.get_noc_addr(s * RT + t), tile);
        noc_async_write(pw_l1 + t * tile, s_pw.get_noc_addr(t * MAX_SLOTS + s), tile);
    }
    noc_async_write_barrier();
}

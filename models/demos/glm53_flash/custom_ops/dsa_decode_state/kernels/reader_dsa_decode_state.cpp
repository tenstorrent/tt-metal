// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the fused DSA decode state maintenance. Role A (tile row tb): the q_a block of the stem and
// the q_a norm weight in chunks. Role B (user b): the index vectors, the user's latent page id, its stem
// blocks (kv_a, indexer k, gate), the constants (once per core), then the cache tiles it updates (the
// latent page, the ring k/g blocks, the pool block) and the placement rows for compute / the writer.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_qa = 0, cb_wq = 1, cb_kva = 2, cb_kblk = 3, cb_gblk = 4, cb_wkva = 5, cb_wk = 6, cb_bk = 7;
constexpr uint32_t cb_ape = 8, cb_sel = 9, cb_lat_old = 10, cb_ring_old = 11, cb_pool_old = 12, cb_selrow = 13;
constexpr uint32_t cb_meta_w = 29, cb_idx = 30;

void kernel_main() {
    constexpr uint32_t Lq_t = get_compile_time_arg_val(0);
    constexpr uint32_t L_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    constexpr uint32_t Ws_t = get_compile_time_arg_val(3);      // stem row width in tiles
    constexpr uint32_t qa_t = get_compile_time_arg_val(4);      // stem column offsets in tiles
    constexpr uint32_t kva_t = get_compile_time_arg_val(5);
    constexpr uint32_t k_t = get_compile_time_arg_val(6);
    constexpr uint32_t g_t = get_compile_time_arg_val(7);
    constexpr uint32_t P = get_compile_time_arg_val(8);         // in-pool positions (ring slots)
    constexpr uint32_t pool_blocks = get_compile_time_arg_val(9);  // pool blocks per user (identity table)
    constexpr uint32_t Bt = get_compile_time_arg_val(10);

    constexpr auto stem_a = TensorAccessorArgs<11>();
    constexpr auto lat_a = TensorAccessorArgs<stem_a.next_compile_time_args_offset()>();
    constexpr auto lpt_a = TensorAccessorArgs<lat_a.next_compile_time_args_offset()>();
    constexpr auto pos_a = TensorAccessorArgs<lpt_a.next_compile_time_args_offset()>();
    constexpr auto rk_a = TensorAccessorArgs<pos_a.next_compile_time_args_offset()>();
    constexpr auto rg_a = TensorAccessorArgs<rk_a.next_compile_time_args_offset()>();
    constexpr auto pool_a = TensorAccessorArgs<rg_a.next_compile_time_args_offset()>();
    constexpr auto wq_a = TensorAccessorArgs<pool_a.next_compile_time_args_offset()>();
    constexpr auto wkva_a = TensorAccessorArgs<wq_a.next_compile_time_args_offset()>();
    constexpr auto wk_a = TensorAccessorArgs<wkva_a.next_compile_time_args_offset()>();
    constexpr auto bk_a = TensorAccessorArgs<wk_a.next_compile_time_args_offset()>();
    constexpr auto ape_a = TensorAccessorArgs<bk_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<ape_a.next_compile_time_args_offset()>();

    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t idx = get_arg_val<uint32_t>(1);  // tb (role A) or user b (role B)
    const uint32_t stem_addr = get_arg_val<uint32_t>(2);
    const uint32_t lat_addr = get_arg_val<uint32_t>(3);
    const uint32_t lpt_addr = get_arg_val<uint32_t>(4);
    const uint32_t pos_addr = get_arg_val<uint32_t>(5);
    const uint32_t rk_addr = get_arg_val<uint32_t>(6);
    const uint32_t rg_addr = get_arg_val<uint32_t>(7);
    const uint32_t pool_addr = get_arg_val<uint32_t>(8);
    const uint32_t wq_addr = get_arg_val<uint32_t>(9);
    const uint32_t wkva_addr = get_arg_val<uint32_t>(10);
    const uint32_t wk_addr = get_arg_val<uint32_t>(11);
    const uint32_t bk_addr = get_arg_val<uint32_t>(12);
    const uint32_t ape_addr = get_arg_val<uint32_t>(13);
    const uint32_t sel_addr = get_arg_val<uint32_t>(14);
    const uint32_t lpt_page_bytes = get_arg_val<uint32_t>(15);  // one page-table row
    const uint32_t pos_page_bytes = get_arg_val<uint32_t>(16);  // the [B] int32 vector

    const uint32_t t32 = get_tile_size(cb_wq);
    const uint32_t t16 = get_tile_size(cb_qa);
    const auto stem_acc = TensorAccessor(stem_a, stem_addr, t16);
    const auto lat_acc = TensorAccessor(lat_a, lat_addr, t16);
    const auto lpt_acc = TensorAccessor(lpt_a, lpt_addr, lpt_page_bytes);
    const auto pos_acc = TensorAccessor(pos_a, pos_addr, pos_page_bytes);
    const auto rk_acc = TensorAccessor(rk_a, rk_addr, t16);
    const auto rg_acc = TensorAccessor(rg_a, rg_addr, t16);
    const auto pool_acc = TensorAccessor(pool_a, pool_addr, t16);
    const auto wq_acc = TensorAccessor(wq_a, wq_addr, t32);
    const auto wkva_acc = TensorAccessor(wkva_a, wkva_addr, t32);
    const auto wk_acc = TensorAccessor(wk_a, wk_addr, t32);
    const auto bk_acc = TensorAccessor(bk_a, bk_addr, t32);
    const auto ape_acc = TensorAccessor(ape_a, ape_addr, t32);
    const auto sel_acc = TensorAccessor(sel_a, sel_addr, t16);

    Noc noc;
    auto read_tiles = [&](const auto& acc, uint32_t cb_id, uint32_t tb, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    read_tiles(sel_acc, cb_sel, t16, 0, 33);

    if (role == 0) {
        // ---- role A: the q_a block of tile row tb, the norm weight streamed 4 tiles at a time
        const uint32_t tb = idx;
        read_tiles(stem_acc, cb_qa, t16, tb * Ws_t + qa_t, Lq_t);
        for (uint32_t c = 0; c < Lq_t; c += 4) {
            read_tiles(wq_acc, cb_wq, t32, c, 4);
        }
        return;
    }

    // ---- role B: user b
    const uint32_t b = idx;
    const uint32_t tb = b >> 5;
    // index vectors: positions[b]; the latent page from the user's page-table row
    CircularBuffer cb_i(cb_idx);
    const uint32_t idx_l1 = cb_i.get_write_ptr();
    noc.async_read(pos_acc, cb_i, pos_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> posv(idx_l1);
    const uint32_t a = posv[b];
    const uint32_t lat_row = a & 31u;
    const uint32_t lat_blk = a >> 5;
    const uint32_t ring_slot = a % P;
    const uint32_t p = a / P;
    const uint32_t pool_row = p & 31u;
    const uint32_t pool_blk = b * pool_blocks + (p >> 5);
    // the page-table entry [b, lat_blk]: a 64 B aligned chunk of the user's row
    const uint32_t ent_off = lat_blk * 4u;
    const uint32_t chunk_off = ent_off & ~63u;
    noc.async_read(lpt_acc, cb_i, 64, {.page_id = b, .offset_bytes = chunk_off}, {.offset_bytes = 256});
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> ptv(idx_l1 + 256);
    const uint32_t lat_page = ptv[(ent_off - chunk_off) >> 2];
    // the selector tiles of the three target rows for compute (it cannot read indices itself) and the page ids
    // for the writer
    {
        CircularBuffer sr(cb_selrow);
        sr.reserve_back(3);
        noc.async_read(sel_acc, sr, t16, {.page_id = lat_row}, {.offset_bytes = 0});
        noc.async_read(sel_acc, sr, t16, {.page_id = ring_slot}, {.offset_bytes = t16});
        noc.async_read(sel_acc, sr, t16, {.page_id = pool_row}, {.offset_bytes = 2 * t16});
        noc.async_read_barrier();
        sr.push_back(3);
        CircularBuffer mw(cb_meta_w);
        mw.reserve_back(1);
        CoreLocalMem<volatile uint32_t> w(mw.get_write_ptr());
        w[0] = lat_page;
        w[1] = b;
        w[2] = pool_blk;
        mw.push_back(1);
    }
    // stem blocks of the user's tile row
    read_tiles(stem_acc, cb_kva, t16, tb * Ws_t + kva_t, L_t);
    read_tiles(stem_acc, cb_kblk, t16, tb * Ws_t + k_t, D_t);
    read_tiles(stem_acc, cb_gblk, t16, tb * Ws_t + g_t, D_t);
    // constants
    read_tiles(wkva_acc, cb_wkva, t32, 0, L_t);
    read_tiles(wk_acc, cb_wk, t32, 0, D_t);
    read_tiles(bk_acc, cb_bk, t32, 0, D_t);
    read_tiles(ape_acc, cb_ape, t32, 0, D_t);
    // cache tiles to update
    read_tiles(lat_acc, cb_lat_old, t16, lat_page * L_t, L_t);
    {
        CircularBuffer cb(cb_ring_old);
        cb.reserve_back(2 * D_t);
        for (uint32_t t = 0; t < D_t; t++) {
            noc.async_read(rk_acc, cb, t16, {.page_id = b * D_t + t}, {.offset_bytes = t * t16});
            noc.async_read(rg_acc, cb, t16, {.page_id = b * D_t + t}, {.offset_bytes = (D_t + t) * t16});
        }
        noc.async_read_barrier();
        cb.push_back(2 * D_t);
    }
    read_tiles(pool_acc, cb_pool_old, t16, pool_blk * D_t, D_t);
    (void)Bt;
}

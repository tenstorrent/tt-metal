// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// K / V rows of one sequence -> its paged BF16 cache, one core (K on BRISC, V on NCRISC), for the DFlash drafter
// (dflash/draft.py).
//
// Row r (r < rows, position pos[r]; pos < 0: skipped) of the K and V head tensors ([1, 32 users, 32 head rows,
// head_dim] BF16, height-sharded one user per core: user r's two 32x32 tiles at the same L1 address on core r, NOC
// coordinates in the runtime args) goes to row pos % 32 of tile ((blk * kv_heads + h) * (block / 32) + (pos % block) /
// 32) * head_tiles + t of the paged cache (blk = page_table[0][pos / block]), for every kv head h and head tile t. The
// valid rows have consecutive positions, so they touch at most two 32-row cache tiles per (h, t): those are read once,
// patched, and written back (whole tiles: DRAM accesses are 64-byte aligned, and one core avoids read-modify-write
// races between rows sharing a tile, which ttnn's paged_update_cache has).
//
// head_offset: the cache's kv head h is head row head_offset + h of the source tiles (several layers' K / V heads in
// one tensor).
//
// KV_WAIT_SEM (with KV_SEL): wait for KV_WAIT_N increments of semaphore KV_WAIT_SEM + KV_SEL (the source shards'
// writers in the same program) before reading the source rows.
//
// KV_SEL = 0 / 1: this kernel writes only K / V (two kernels on BRISC and NCRISC of the same core, each with its own
// scratch header).
//
// KV_NOTIFY_N: signal the fused attention cores once the source shards are written (before this kernel reads them:
// the attention cores patch the block's rows in from the shards).
//
// runtime args: [k_src, v_src, k_cache, v_cache, pos, page_table, (noc_x, noc_y) x rows]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t k_src = get_arg_val<uint32_t>(0);
    const uint32_t v_src = get_arg_val<uint32_t>(1);
    const uint32_t k_cache = get_arg_val<uint32_t>(2);
    const uint32_t v_cache = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t pt_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t kv_heads = get_compile_time_arg_val(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t head_offset = get_compile_time_arg_val(6);
    constexpr auto kc_args = TensorAccessorArgs<7>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile = 2048;
    const auto s_pos = TensorAccessor(pos_args, pos_addr, 128);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_row_bytes);

    cb_reserve_back(cb_scr, 1);
    constexpr uint32_t kv_lo = KV_SEL, kv_hi = KV_SEL + 1;
    // scratch: [K kernel's header (256 bytes) | V kernel's header (256 bytes) | dst: the landed cache tiles, per layer
    // (kv, group, h, t)]; a header holds the positions (128 bytes) and two 64-byte page-table reads
    const uint32_t scr = get_write_ptr(cb_scr) + KV_SEL * 256;
    const uint32_t dst = get_write_ptr(cb_scr) + 512;

    noc_async_read(s_pos.get_noc_addr(0), scr, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    int32_t pos[rows];
    uint32_t first = rows, last = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        pos[r] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[r];
        if (pos[r] >= 0) {
            first = first == rows ? r : first;
            last = r;
        }
    }
    if (first == rows) {
        return;
    }
    const uint32_t p0 = pos[first], p1 = pos[last];
    const uint32_t bi0 = p0 / block, bi1 = p1 / block;
    noc_async_read(s_pt.get_noc_addr(0) + ((bi0 * 4) & ~63u), scr + 128, 64);
    noc_async_read(s_pt.get_noc_addr(0) + ((bi1 * 4) & ~63u), scr + 192, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t blk0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128)[bi0 & 15];
    const uint32_t blk1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 192)[bi1 & 15];
    const uint32_t tr0 = (p0 % block) / 32, tr1 = (p1 % block) / 32;
    const uint32_t groups = (bi0 != bi1 || tr0 != tr1) ? 2 : 1;
#ifdef KV_LAYERS
    // KV_LAYERS = L (the drafter's context pass): the same source rows go to L layers' caches, layer l's K / V heads
    // being source head rows head_offset + l kv_heads (runtime args 6 + 2 rows + 2 (l - 1): layer l >= 1's caches);
    // every layer's cache tiles are read at once and written back at once
    constexpr uint32_t n_layers = KV_LAYERS;
#else
    constexpr uint32_t n_layers = 1;
#endif
    auto cache_tile = [&](uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        const uint32_t blk = g == 0 ? blk0 : blk1, tr = g == 0 ? tr0 : tr1;
        return ((blk * kv_heads + h) * (block / 32) + tr) * head_tiles + t;
    };
    auto dst_l1 = [&](uint32_t l, uint32_t kv, uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        return dst + ((((l * 2 + kv) * 2 + g) * kv_heads + h) * head_tiles + t) * tile;
    };
    auto layer_cache = [&](uint32_t l, uint32_t kv) -> uint32_t {
        if (l == 0) {
            return kv == 0 ? k_cache : v_cache;
        }
        return get_arg_val<uint32_t>(6 + 2 * rows + 2 * (l - 1) + kv);
    };
    for (uint32_t l = 0; l < n_layers; ++l) {
        const auto s_k = TensorAccessor(kc_args, layer_cache(l, 0), tile);
        const auto s_v = TensorAccessor(vc_args, layer_cache(l, 1), tile);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t h = 0; h < kv_heads; ++h) {
                for (uint32_t t = 0; t < head_tiles; ++t) {
                    if (kv_lo == 0) {
                        noc_async_read(s_k.get_noc_addr(cache_tile(g, h, t)), dst_l1(l, 0, g, h, t), tile);
                    }
                    if (kv_hi == 2) {
                        noc_async_read(s_v.get_noc_addr(cache_tile(g, h, t)), dst_l1(l, 1, g, h, t), tile);
                    }
                }
            }
        }
    }
#ifdef KV_WAIT_SEM
    // fused after dflash/qk_prep.py: each prep core bumps semaphore KV_WAIT_SEM + kv once its K / V shard is written
    {
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(KV_WAIT_SEM + kv_lo));
        noc_semaphore_wait(sem, KV_WAIT_N);
        noc_semaphore_set(sem, 0);
    }
#endif
#ifdef KV_NOTIFY_N
    // the attention cores patch the block's rows into their own copies of the cache tiles (from the
    // source shards, kernels/ba_reader.cpp BA_PATCH): signal them once the source shards are written, not after the
    // cache write
    for (uint32_t i = 0; i < KV_NOTIFY_N; ++i) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(KV_NOTIFY_ARG + 2 * i),
                get_arg_val<uint32_t>(KV_NOTIFY_ARG + 2 * i + 1),
                get_semaphore(KV_NOTIFY_SEM)),
            1);
    }
    noc_async_atomic_barrier();
#endif
    // Each valid row's 32-byte face rows straight from its user core into the landed cache tiles (row pos % 32):
    // no staging of the whole source tiles, no copy loop.
    noc_async_read_barrier();  // the cache tiles (their other rows are kept)
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (uint32_t r = first; r <= last; ++r) {
            if (pos[r] < 0) {
                continue;
            }
            const uint32_t x = get_arg_val<uint32_t>(6 + 2 * r), y = get_arg_val<uint32_t>(7 + 2 * r);
            const uint32_t p = pos[r];
            const uint32_t g = (p / block == bi0 && (p % block) / 32 == tr0) ? 0 : 1;
            const uint32_t rr = p % 32;
            const uint32_t dst_off = (rr < 16 ? 0 : 1024) + (rr % 16) * 32;
            for (uint32_t kv = kv_lo; kv < kv_hi; ++kv) {
                for (uint32_t h = 0; h < kv_heads; ++h) {
                    const uint32_t hs = head_offset + l * kv_heads + h;
                    const uint32_t src_off = (hs < 16 ? 0 : 1024) + (hs % 16) * 32;
                    for (uint32_t t = 0; t < head_tiles; ++t) {
                        const uint32_t s_ = (kv == 0 ? k_src : v_src) + t * tile + src_off;
                        const uint32_t d = dst_l1(l, kv, g, h, t) + dst_off;
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(get_noc_addr(x, y, s_ + f * 512), d + f * 512, 32);
                        }
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t l = 0; l < n_layers; ++l) {
        const auto s_k = TensorAccessor(kc_args, layer_cache(l, 0), tile);
        const auto s_v = TensorAccessor(vc_args, layer_cache(l, 1), tile);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t h = 0; h < kv_heads; ++h) {
                for (uint32_t t = 0; t < head_tiles; ++t) {
                    if (kv_lo == 0) {
                        noc_async_write(dst_l1(l, 0, g, h, t), s_k.get_noc_addr(cache_tile(g, h, t)), tile);
                    }
                    if (kv_hi == 2) {
                        noc_async_write(dst_l1(l, 1, g, h, t), s_v.get_noc_addr(cache_tile(g, h, t)), tile);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}

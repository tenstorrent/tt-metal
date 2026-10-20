// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash drafter (dflash/batched.py BatchedKVWrite): every user's block rows -> its paged BF16 cache, one
// core per user (K on BRISC, V on NCRISC), for one or several layers (KV_LAYERS).
//
// The source K / V are the drafter's rotated heads in the prefill layout [1, kv_heads, rows, 64] (BF16 tile, L1
// interleaved: 16-byte aligned face rows; row R = 32 i + u is block row i of user u). User u's block row i (position
// pos[M u + i], user-major INT32 positions; < 0: skipped) goes to row pos % 32 of tile
// ((blk * kv_heads + h) * (block / 32) + (pos % block) / 32) * head_tiles + t of the cache (blk = page_table[u][pos /
// block]). The M rows of a block have consecutive positions, so they touch at most two 32-row cache tiles per (h, t):
// read once, patched, written back whole (DRAM accesses are 64-byte aligned; one core per user: no read-modify-write
// races between rows sharing a tile).
//
// runtime args: [user, k_src, v_src, k_cache, v_cache, pos, page_table, (k_cache, v_cache) of layers 1 .. L - 1]
// compile-time args: [cb_scr, M, kv_heads, head_tiles, block, pt_row_bytes, users, accessors (k_cache, v_cache, pos,
// page_table, k_src)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t user = get_arg_val<uint32_t>(0);
    const uint32_t k_src = get_arg_val<uint32_t>(1);
    const uint32_t v_src = get_arg_val<uint32_t>(2);
    const uint32_t k_cache = get_arg_val<uint32_t>(3);
    const uint32_t v_cache = get_arg_val<uint32_t>(4);
    const uint32_t pos_addr = get_arg_val<uint32_t>(5);
    const uint32_t pt_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t M = get_compile_time_arg_val(1);
    constexpr uint32_t kv_heads = get_compile_time_arg_val(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t users = get_compile_time_arg_val(6);
    constexpr auto kc_args = TensorAccessorArgs<7>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
#ifdef KV_LAYERS
    constexpr uint32_t n_layers = KV_LAYERS;
#else
    constexpr uint32_t n_layers = 1;
#endif

    constexpr uint32_t tile = 2048;
    constexpr uint32_t rows = M * users;       // source rows
    constexpr uint32_t row_tiles = rows / 32;  // source tile rows per head
    const auto s_pos = TensorAccessor(pos_args, pos_addr, rows * 4);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_row_bytes);

    cb_reserve_back(cb_scr, 1);
    constexpr uint32_t kv_lo = KV_SEL, kv_hi = KV_SEL + 1;
    // scratch: [K kernel's header (256 bytes) | V kernel's header (256 bytes) | the landed cache tiles (layer, kv,
    // group, h, t)]; a header holds the positions (64 bytes from an aligned window) and two 64-byte page-table reads
    const uint32_t scr = get_write_ptr(cb_scr) + KV_SEL * 256;
    const uint32_t dst = get_write_ptr(cb_scr) + 512;

    const uint32_t pos_off = user * M * 4, pos_a = pos_off & ~63u;
    noc_async_read(s_pos.get_noc_addr(0) + pos_a, scr, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    int32_t pos[M];
    uint32_t first = M, last = 0;
    for (uint32_t r = 0; r < M; ++r) {
        pos[r] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + (pos_off - pos_a))[r];
        if (pos[r] >= 0) {
            first = first == M ? r : first;
            last = r;
        }
    }
    if (first == M) {
        return;
    }
    const uint32_t p0 = pos[first], p1 = pos[last];
    const uint32_t bi0 = p0 / block, bi1 = p1 / block;
    noc_async_read(s_pt.get_noc_addr(user) + ((bi0 * 4) & ~63u), scr + 128, 64);
    noc_async_read(s_pt.get_noc_addr(user) + ((bi1 * 4) & ~63u), scr + 192, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t blk0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128)[bi0 & 15];
    const uint32_t blk1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 192)[bi1 & 15];
    const uint32_t tr0 = (p0 % block) / 32, tr1 = (p1 % block) / 32;
    const uint32_t groups = (bi0 != bi1 || tr0 != tr1) ? 2 : 1;

    auto cache_tile = [&](uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        const uint32_t blk = g == 0 ? blk0 : blk1, tr = g == 0 ? tr0 : tr1;
        return ((blk * kv_heads + h) * (block / 32) + tr) * head_tiles + t;
    };
    auto dst_l1 = [&](uint32_t l, uint32_t kv, uint32_t g, uint32_t h, uint32_t t) -> uint32_t {
        return dst + ((((l * 2 + kv) * 2 + g) * kv_heads + h) * head_tiles + t) * tile;
    };
    auto layer_arg = [&](uint32_t l, uint32_t kv, uint32_t first_arg) -> uint32_t {
        // layer l's (k, v) cache / source: runtime args first_arg + 4 (l - 1) + kv (l >= 1)
        return get_arg_val<uint32_t>(first_arg + 4 * (l - 1) + kv);
    };
    auto cache_addr = [&](uint32_t l, uint32_t kv) -> uint32_t {
        return l == 0 ? (kv == 0 ? k_cache : v_cache) : layer_arg(l, kv, 7);
    };
    auto src_addr = [&](uint32_t l, uint32_t kv) -> uint32_t {
        return l == 0 ? (kv == 0 ? k_src : v_src) : layer_arg(l, kv, 9);
    };
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (uint32_t kv = kv_lo; kv < kv_hi; ++kv) {
            const auto s_c = TensorAccessor(kc_args, cache_addr(l, kv), tile);
            for (uint32_t g = 0; g < groups; ++g) {
                for (uint32_t h = 0; h < kv_heads; ++h) {
                    for (uint32_t t = 0; t < head_tiles; ++t) {
                        noc_async_read(s_c.get_noc_addr(cache_tile(g, h, t)), dst_l1(l, kv, g, h, t), tile);
                    }
                }
            }
        }
    }
    noc_async_read_barrier();  // the cache tiles (their other rows are kept)
    // each valid row's two 32-byte face rows from the source tiles into the landed cache tiles (row pos % 32)
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (uint32_t kv = kv_lo; kv < kv_hi; ++kv) {
            const auto s_s = TensorAccessor(src_args, src_addr(l, kv), tile);
            for (uint32_t r = first; r <= last; ++r) {
                if (pos[r] < 0) {
                    continue;
                }
                const uint32_t p = pos[r];
                const uint32_t g = (p / block == bi0 && (p % block) / 32 == tr0) ? 0 : 1;
                const uint32_t rr = p % 32;
                const uint32_t dst_off = (rr < 16 ? 0 : 1024) + (rr % 16) * 32;
                const uint32_t R = r * users + user;  // the source row
                const uint32_t sr = R % 32;
                const uint32_t src_off = (sr < 16 ? 0 : 1024) + (sr % 16) * 32;
                for (uint32_t h = 0; h < kv_heads; ++h) {
                    for (uint32_t t = 0; t < head_tiles; ++t) {
                        const uint64_t s_ = s_s.get_noc_addr((h * row_tiles + R / 32) * head_tiles + t) + src_off;
                        const uint32_t d = dst_l1(l, kv, g, h, t) + dst_off;
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(s_ + f * 512, d + f * 512, 32);
                        }
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (uint32_t kv = kv_lo; kv < kv_hi; ++kv) {
            const auto s_c = TensorAccessor(kc_args, cache_addr(l, kv), tile);
            for (uint32_t g = 0; g < groups; ++g) {
                for (uint32_t h = 0; h < kv_heads; ++h) {
                    for (uint32_t t = 0; t < head_tiles; ++t) {
                        noc_async_write(dst_l1(l, kv, g, h, t), s_c.get_noc_addr(cache_tile(g, h, t)), tile);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}

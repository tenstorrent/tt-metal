// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill per-head RMSNorm + partial RoPE (qk_rope.py), writer (BRISC): per item the compute's tiles 2 .. Dt - 1, then
// 0 and 1, to the head-major output.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, items, ncores, grid_x, H, Mt, Dt
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first, n_items;
    core_span(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        1,
        first,
        n_items);
    const uint32_t H = get_common_arg_val<uint32_t>(4), Mt = get_common_arg_val<uint32_t>(5);
    const uint32_t Dt = get_common_arg_val<uint32_t>(6);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16;
    const uint32_t TB = get_tile_size(cb_out);
    const auto os = TensorAccessor(o_args, o_addr, TB);
#if defined(KV_CACHE)
    // Decode (T = 1) straight into the bf8 paged K / V caches [blocks, H, bs, D] (qk_rope.py cache=): per item (head h)
    // the compute's Dt K tiles (order 2 .. Dt - 1, 0, 1) then Dt V tiles, packed bfp8. Only their row 0 is the token:
    // its 2 shared exponents (bytes 0, 16) and 2 x 16 mantissa bytes (64, 320) are written into row r of the cache
    // tile at position pos (page_table[pos / bs]), byte-granular, in place (no read-modify-write of the tile).
    const uint32_t vc = get_common_arg_val<uint32_t>(8);  // (7: the K cache, o_addr)
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(9), pt_addr = get_common_arg_val<uint32_t>(10);
    const uint32_t bs = get_common_arg_val<uint32_t>(11);
    constexpr auto vc_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto& kcs = os;
    const auto vcs = TensorAccessor(vc_args, vc, TB);
    const auto poss = TensorAccessor(pos_args, pos_addr, 4);
    const auto pts = TensorAccessor(pt_args, pt_addr, 4);
    constexpr uint32_t cb_v = 17;
    cb_reserve_back(cb_v, 1);
    const uint32_t scratch = (get_write_ptr(cb_v) + 63) & ~63u;  // 64-B aligned, as the 64-B DRAM reads below
    noc_async_read(poss.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    const uint32_t pos = *(volatile tt_l1_ptr uint32_t*)scratch;
    const uint32_t bi = (pos / bs) * 4;
    noc_async_read(pts.get_noc_addr(0) + (bi & ~63u), scratch, 64);
    noc_async_read_barrier();
    const uint32_t blk = *(volatile tt_l1_ptr uint32_t*)(scratch + (bi & 63));
    const uint32_t row = pos % bs, rt = row / 32, rr = row % 32;
    const uint32_t fa = rr < 16 ? 0 : 2, fr = rr % 16;
    const uint32_t e0 = fa * 16 + fr, e1 = e0 + 16, m0 = 64 + fa * 256 + fr * 16, m1 = m0 + 256;
#ifdef KV_L1
    // the same row into the decode SDPA's prefetched K / V (sdpa_dec/kernels/kv_prefetch.hpp, already in L1 when this
    // op runs): on core g NG + c holding tile row pos / 32 (its c-th share), at its slot there
    const uint32_t kvbuf = get_common_arg_val<uint32_t>(12), maxr = get_common_arg_val<uint32_t>(13);
    const uint32_t NG = get_common_arg_val<uint32_t>(14);
    const uint32_t nt = (pos + 32) / 32, tl = pos / 32;
    uint32_t oc = 0;
    while ((oc + 1) * nt / NG <= tl) {
        ++oc;
    }
    const uint32_t slot = tl - oc * nt / NG;
#endif
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t h = it % H;
        const uint32_t base = ((blk * H + h) * (bs / 32) + rt) * Dt;
#ifdef KV_L1
        const uint32_t ov = get_common_arg_val<uint32_t>(15 + h * NG + oc);
        const uint64_t own = get_noc_addr(ov >> 16, ov & 0xffff, kvbuf);
#endif
        for (uint32_t k = 0; k < 2 * Dt; ++k) {
            const uint32_t j = k >= Dt ? k - Dt : (k < Dt - 2 ? k + 2 : k - (Dt - 2));
            cb_wait_front(cb_out, 1);
            const uint32_t t = get_read_ptr(cb_out);
            volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)t;  // (word access only)
            uint32_t mv[8];
            for (uint32_t i = 0; i < 4; ++i) {
                mv[i] = w[16 + i];      // face 0 (or 2) row 0 mantissas, bytes 64 ..
                mv[4 + i] = w[80 + i];  // face 1 row 0, bytes 320 ..
            }
            const uint32_t x0 = w[0] & 0xff, x1 = w[4] & 0xff;
            w[e0 / 4] = (w[e0 / 4] & ~(0xffu << (8 * (e0 % 4)))) | (x0 << (8 * (e0 % 4)));
            w[e1 / 4] = (w[e1 / 4] & ~(0xffu << (8 * (e1 % 4)))) | (x1 << (8 * (e1 % 4)));
            for (uint32_t i = 0; i < 4; ++i) {
                w[m0 / 4 + i] = mv[i];
                w[m1 / 4 + i] = mv[4 + i];
            }
            const uint64_t d = k < Dt ? kcs.get_noc_addr(base + j) : vcs.get_noc_addr(base + j);
            noc_async_write(t + e0, d + e0, 1);
            noc_async_write(t + e1, d + e1, 1);
            noc_async_write(t + m0, d + m0, 16);
            noc_async_write(t + m1, d + m1, 16);
#ifdef KV_L1
            const uint64_t l = own + ((k < Dt ? slot : maxr + slot) * Dt + j) * TB;
            noc_async_write(t + e0, l + e0, 1);
            noc_async_write(t + e1, l + e1, 1);
            noc_async_write(t + m0, l + m0, 16);
            noc_async_write(t + m1, l + m1, 16);
#endif
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    (void)Mt;
#elif defined(TOKEN_MAJOR)
    // Decode (T = 1): row 0 of head h's tiles goes to row h of a token-major [32, Dt * 32] L1 shard on one core (the
    // paged cache update's input, qk_rope.py kv_out), and head h of V (head-major, row 0 of its tiles) likewise into
    // a second one. Rows are 32 + 32 B (face 0 / face 1, or faces 2 / 3 for h >= 16).
    const uint32_t sh_x = get_common_arg_val<uint32_t>(7), sh_y = get_common_arg_val<uint32_t>(8);
    const uint32_t k_sh = get_common_arg_val<uint32_t>(9), v_sh = get_common_arg_val<uint32_t>(10);
    constexpr auto v_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto vs = TensorAccessor(v_args, get_common_arg_val<uint32_t>(11), TB);
    constexpr uint32_t cb_v = 17;
    cb_reserve_back(cb_v, 1);
    const uint32_t scratch = get_write_ptr(cb_v);
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t h = it % H;
#ifdef Q_HUB
        // the decode SDPA's q (sdpa_dec): head h at row h % 6 of kv group h / 6's hub core (x << 16 | y, args 12 ..)
        const uint32_t hub = get_common_arg_val<uint32_t>(12 + h / 6);
        const uint32_t hx = hub >> 16, hy = hub & 0xffff, ro = (h % 6) * 32;
#else
        const uint32_t hx = sh_x, hy = sh_y, ro = (h < 16 ? 0 : 1024) + (h % 16) * 32;
#endif
        for (uint32_t k = 0; k < Dt; ++k) {
            const uint32_t j = k < Dt - 2 ? k + 2 : k - (Dt - 2);
            cb_wait_front(cb_out, 1);
            const uint64_t d = get_noc_addr(hx, hy, k_sh + j * TB + ro);
            noc_async_write(get_read_ptr(cb_out), d, 32);
            noc_async_write(get_read_ptr(cb_out) + 512, d + 512, 32);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
        for (uint32_t j = 0; v_sh != 0 && j < Dt; ++j) {  // v_sh 0: no V (q)
            const uint64_t s = vs.get_noc_addr(h * Dt + j);
            noc_async_read(s, scratch, 64);  // face 0 rows 0-1 (64-B DRAM reads)
            noc_async_read(s + 512, scratch + 64, 64);
            noc_async_read_barrier();
            const uint64_t d = get_noc_addr(sh_x, sh_y, v_sh + j * TB + ro);
            noc_async_write(scratch, d, 32);
            noc_async_write(scratch + 64, d + 512, 32);
            noc_async_writes_flushed();
        }
    }
    (void)os;
    (void)Mt;
#else
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t r = it / H, h = it % H, base = (h * Mt + r) * Dt;
        for (uint32_t k = 0; k < Dt; ++k) {
            const uint32_t j = k < Dt - 2 ? k + 2 : k - (Dt - 2);
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(base + j), TB);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
#endif
    noc_async_write_barrier();
}

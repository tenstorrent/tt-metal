// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode SDPA (sdpa_dec.py), reader (NCRISC). Core (g, c): kv head g, the c-th of NG spans of the nt = ceil(n / 32)
// position tile rows (n = cur_pos + 1). Reads cur_pos and the page table, then in one batch: its group's 6 q heads
// (rows 6g .. 6g + 5 of the token-major q, into rows 0 .. 5), the constant tiles and its K / V tile rows (bf8 paged
// caches [blocks, H, bs, D]). Publishes (nr, last, valid columns of the last tile) to the compute and writer (cb_info).
#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // common args: pos, page table, K cache, V cache, q, consts, bs, page table bytes, NG, H
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0), pt_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(2), v_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(4), c_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t bs = get_common_arg_val<uint32_t>(6), pt_bytes = get_common_arg_val<uint32_t>(7);
    const uint32_t NG = get_common_arg_val<uint32_t>(8), H = get_common_arg_val<uint32_t>(9);
    const uint32_t g = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1);
    constexpr auto pos_args = TensorAccessorArgs<0>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_one = 3, cb_info = 5, cb_tmp = 6;
    constexpr uint32_t Dt = 8;
    const uint32_t KB = get_tile_size(cb_k), QB = get_tile_size(cb_q);
    const auto poss = TensorAccessor(pos_args, pos_addr, 4);
    const auto pts = TensorAccessor(pt_args, pt_addr, pt_bytes);
    const auto ks = TensorAccessor(k_args, k_addr, KB);
    const auto vs = TensorAccessor(v_args, v_addr, KB);
    const auto qs = TensorAccessor(q_args, q_addr, QB);
    (void)c_addr;

    uint32_t nr, r0, r1;
    volatile tt_l1_ptr uint32_t* pt;
    {
#ifdef SDPA_ZONES
        DeviceZoneScopedN("TD-R-meta");
#endif
#ifdef KV_PF
        // K / V already in this core's L1 (kv_prefetch.hpp), its share in its record
        const uint32_t tmp = get_common_arg_val<uint32_t>(10);
        (void)poss;
        (void)pts;
#else
        const uint32_t tmp = (get_write_ptr(cb_tmp) + 63) & ~63u;
        noc_async_read(poss.get_noc_addr(0), tmp, 64);
        noc_async_read(pts.get_noc_addr(0), tmp + 64, (pt_bytes + 63) & ~63u);
#endif
        // q while those land (the constant tiles: the writer)
        cb_reserve_back(cb_q, Dt);
        const uint32_t lq = get_write_ptr(cb_q);
#ifdef Q_HUB
        // rows 0 .. 5 of this group's hub (qk_rope.py q_hub): 192 B per face
        const uint32_t hub = get_common_arg_val<uint32_t>(11 + g);
        for (uint32_t d = 0; d < Dt; ++d) {
            const uint64_t src = get_noc_addr(hub >> 16, hub & 0xffff, q_addr + d * QB);
            noc_async_read(src, lq + d * QB, 192);
            noc_async_read(src + 512, lq + d * QB + 512, 192);
        }
        (void)qs;
#else
        for (uint32_t d = 0; d < Dt; ++d) {
            const uint64_t src = qs.get_noc_addr(d);
            for (uint32_t r = 0; r < 6;) {  // contiguous row runs (split at face row 16)
                const uint32_t h = 6 * g + r;
                const uint32_t run = h < 16 ? (16 - h < 6 - r ? 16 - h : 6 - r) : 6 - r;
                const uint32_t so = h < 16 ? h * 32 : 1024 + (h - 16) * 32;
                noc_async_read(src + so, lq + d * QB + r * 32, run * 32);
                noc_async_read(src + so + 512, lq + d * QB + 512 + r * 32, run * 32);
                r += run;
            }
        }
#endif
        noc_async_read_barrier();
        const uint32_t n = *(volatile tt_l1_ptr uint32_t*)tmp + 1;
        pt = (volatile tt_l1_ptr uint32_t*)(tmp + 64);
        const uint32_t nt = (n + 31) / 32;
        r0 = c * nt / NG, r1 = (c + 1) * nt / NG, nr = r1 - r0;
        cb_reserve_back(cb_info, 1);
        volatile tt_l1_ptr uint32_t* info = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_info);
        info[0] = nr;
        info[1] = nr != 0 && r1 == nt;
        info[2] = n - (nt - 1) * 32;
#ifdef KV_PF
        nr = ((volatile tt_l1_ptr uint32_t*)tmp)[1];  // the prefetch's share (the same split)
        info[0] = nr;
#endif
        cb_push_back(cb_info, 1);
        cb_push_back(cb_q, Dt);
    }
    if (nr == 0) {
        return;
    }
#ifdef SDPA_ZONES
    DeviceZoneScopedN("TD-R-kv");
#endif
#ifdef KV_PF
    cb_reserve_back(cb_k, nr * Dt);
    cb_reserve_back(cb_v, nr * Dt);
    cb_push_back(cb_k, nr * Dt);
    cb_push_back(cb_v, nr * Dt);
    return;
#endif
    cb_reserve_back(cb_k, nr * Dt);
    cb_reserve_back(cb_v, nr * Dt);
    const uint32_t lk = get_write_ptr(cb_k), lv = get_write_ptr(cb_v);
    for (uint32_t t = r0; t < r1; ++t) {
        const uint32_t p0 = 32 * t;
        const uint32_t base = ((pt[p0 / bs] * H + g) * (bs / 32) + (p0 % bs) / 32) * Dt;
        for (uint32_t d = 0; d < Dt; ++d) {
            noc_async_read(ks.get_noc_addr(base + d), lk + ((t - r0) * Dt + d) * KB, KB);
        }
        for (uint32_t d = 0; d < Dt; ++d) {
            noc_async_read(vs.get_noc_addr(base + d), lv + ((t - r0) * Dt + d) * KB, KB);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, nr * Dt);
    cb_push_back(cb_v, nr * Dt);
}

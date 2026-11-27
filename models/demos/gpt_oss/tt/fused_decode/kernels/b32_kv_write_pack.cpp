// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched DFlash verify (dflash/batched.py BatchedKVWrite, BFP8_b caches, one layer, one core per (user, kv head)):
// the block rows into the users' paged BFP8_b caches with the packing on the compute core
// (b32_kv_write_pack_compute.cpp) instead of the RISC-V (b32_kv_write.cpp KV_BF8 spent ~15 of its ~20 us packing rows
// there). K on BRISC, V on NCRISC.
//
// Per kv: the (at most 2 groups x head_tiles) cache tiles the user's M rows touch (BFP8_b) -> cb_old; a BF16 tile per
// cache tile holding just the new rows (zero elsewhere; row pos % 32, from the source's prefill-layout heads) ->
// cb_new; BRISC also the groups' keep masks (BF16: 1.0, 0 on the new rows) -> cb_mask and the group count to the
// compute (mailbox). The compute packs old * mask + new to BFP8_b (the packer: a row's 16-value halves share an
// exponent, so the kept rows pack to the same bytes; the new rows round half up as paged_fused_update_cache's) ->
// cb_out, written back to the cache.
//
// runtime args: [p0 | p1 << 16, k_src, v_src, k_cache, v_cache, pos, page_table, (KV_QK_JOINT: v_src of V head 1)]: the
// core's (user, kv head) pairs p = user kv_heads + h in [p0, p1), one after another
// compile-time args: [cb_scr, M, kv_heads, head_tiles, block, pt_row_bytes, users, cb_old, cb_new, cb_mask, cb_out,
// accessors (k_cache, v_cache, pos, page_table, k_src)]  (cb_old / cb_new / cb_out: K's; V's are the next index)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t arg0 = get_arg_val<uint32_t>(0);
    const uint32_t pair0 = arg0 & 0xFFFF, pair1 = arg0 >> 16;
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    cb_reserve_back(cb_scr, 1);
    for (uint32_t pr = pair0; pr < pair1; ++pr) {
        const uint32_t user = pr / get_compile_time_arg_val(2), h = pr % get_compile_time_arg_val(2);
#ifdef KV_QK_JOINT
    // the sources as split by nlp_create_qkv_heads(num_heads = q heads + kv heads, num_kv_heads = 1) (one RoPE op
    // for Q and K): K = heads KV_QK_JOINT + h of the "q" tensor (arg 1), V head 0 = the "k" tensor (arg 2), V head 1 =
    // the "v" tensor (arg 7)
#if KV_SEL == 0
    const uint32_t src_addr = get_arg_val<uint32_t>(1), hs = h + KV_QK_JOINT;
#else
    const uint32_t src_addr = h == 0 ? get_arg_val<uint32_t>(2) : get_arg_val<uint32_t>(7), hs = 0;
#endif
#else
    const uint32_t src_addr = get_arg_val<uint32_t>(1 + KV_SEL), hs = h;
#endif
    const uint32_t cache_addr = get_arg_val<uint32_t>(3 + KV_SEL);
    const uint32_t pos_addr = get_arg_val<uint32_t>(5);
    const uint32_t pt_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t M = get_compile_time_arg_val(1);
    constexpr uint32_t kv_heads = get_compile_time_arg_val(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t users = get_compile_time_arg_val(6);
    constexpr uint32_t cb_old = get_compile_time_arg_val(7) + KV_SEL;
    constexpr uint32_t cb_new = get_compile_time_arg_val(8) + KV_SEL;
    constexpr uint32_t cb_mask = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10) + KV_SEL;
    constexpr auto kc_args = TensorAccessorArgs<11>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile8 = 1088, tile16 = 2048;
    constexpr uint32_t rows = M * users;
    constexpr uint32_t row_tiles = rows / 32;
    const auto s_pos = TensorAccessor(pos_args, pos_addr, rows * 4);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_row_bytes);
    const auto s_c = TensorAccessor(kc_args, cache_addr, tile8);  // (the K / V caches share one layout)
    const auto s_s = TensorAccessor(src_args, src_addr, tile16);

    const uint32_t scr = get_write_ptr(cb_scr) + KV_SEL * 256;
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
    uint32_t groups = 0, blk0 = 0, blk1 = 0, tr0 = 0, tr1 = 0, bi0 = 0;
    if (first < M) {
        const uint32_t p0 = pos[first], p1 = pos[last];
        bi0 = p0 / block;
        const uint32_t bi1 = p1 / block;
        noc_async_read(s_pt.get_noc_addr(user) + ((bi0 * 4) & ~63u), scr + 128, 64);
        noc_async_read(s_pt.get_noc_addr(user) + ((bi1 * 4) & ~63u), scr + 192, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        blk0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128)[bi0 & 15];
        blk1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 192)[bi1 & 15];
        tr0 = (p0 % block) / 32;
        tr1 = (p1 % block) / 32;
        groups = (bi0 != bi1 || tr0 != tr1) ? 2 : 1;
    }
#if KV_SEL == 0
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, groups);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, groups);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, groups);
#endif
    if (groups == 0) {
        continue;
    }
    const uint32_t n = groups * head_tiles;
    // cb_old / cb_new / cb_mask / cb_out hold one pair's maximum (2 groups) and every pair moves that many pages, so
    // each pair starts at their first page (their tiles are addressed linearly from the pointers)
    constexpr uint32_t n_max = 2 * head_tiles;
    auto cache_tile = [&](uint32_t g, uint32_t t) -> uint32_t {
        const uint32_t blk = g == 0 ? blk0 : blk1, tr = g == 0 ? tr0 : tr1;
        return ((blk * kv_heads + h) * (block / 32) + tr) * head_tiles + t;
    };
    auto group_of = [&](uint32_t p) -> uint32_t { return (p / block == bi0 && (p % block) / 32 == tr0) ? 0 : 1; };

    // the old cache tiles, tile q = g head_tiles + t
    cb_reserve_back(cb_old, n_max);
    const uint32_t old_l1 = get_write_ptr(cb_old);
    for (uint32_t q = 0; q < n; ++q) {
        noc_async_read(s_c.get_noc_addr(cache_tile(q / head_tiles, q % head_tiles)), old_l1 + q * tile8, tile8);
    }
    // the new rows' tiles: zeros, then each row's two 32-byte face rows at row pos % 32
    cb_reserve_back(cb_new, n_max);
    const uint32_t new_l1 = get_write_ptr(cb_new);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < n * tile16; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, new_l1 + z, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    for (uint32_t r = first; r <= last; ++r) {
        if (pos[r] < 0) {
            continue;
        }
        const uint32_t p = pos[r], g = group_of(p), rr = p % 32;
        const uint32_t dst_off = (rr < 16 ? 0 : 1024) + (rr % 16) * 32;
        const uint32_t R = r * users + user, sr = R % 32;
        const uint32_t src_off = (sr < 16 ? 0 : 1024) + (sr % 16) * 32;
        for (uint32_t t = 0; t < head_tiles; ++t) {
            const uint64_t s_ = s_s.get_noc_addr((hs * row_tiles + R / 32) * head_tiles + t) + src_off;
            const uint32_t d = new_l1 + (g * head_tiles + t) * tile16 + dst_off;
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(s_ + f * 512, d + f * 512, 32);
            }
        }
    }
#if KV_SEL == 0
    // the groups' keep masks: 1.0 everywhere but the new rows
    cb_reserve_back(cb_mask, 2);
    const uint32_t mask_l1 = get_write_ptr(cb_mask);
    for (uint32_t g = 0; g < groups; ++g) {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1 + g * tile16);
        for (uint32_t i = 0; i < tile16 / 4; ++i) {
            w[i] = 0x3F803F80u;
        }
    }
    for (uint32_t r = first; r <= last; ++r) {
        if (pos[r] < 0) {
            continue;
        }
        const uint32_t p = pos[r], g = group_of(p), rr = p % 32;
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                mask_l1 + g * tile16 + (rr < 16 ? 0 : 1024) + f * 512 + (rr % 16) * 32);
            for (uint32_t i = 0; i < 8; ++i) {
                w[i] = 0;
            }
        }
    }
    cb_push_back(cb_mask, 2);
#endif
    noc_async_read_barrier();
    cb_push_back(cb_old, n_max);
    cb_push_back(cb_new, n_max);

    // the packed tiles back to the cache (cb_out: n_max pages per pair, its first n packed)
    cb_wait_front(cb_out, n_max);
    const uint32_t out_l1 = get_read_ptr(cb_out);
    for (uint32_t q = 0; q < n; ++q) {
        noc_async_write(out_l1 + q * tile8, s_c.get_noc_addr(cache_tile(q / head_tiles, q % head_tiles)), tile8);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_max);
    }
}

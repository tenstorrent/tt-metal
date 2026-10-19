// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + output scatter (BRISC, NOC1) for the decode streamed linear op (fused_decode/stream.py:
// LinearStream).
//
// 1. Gathers the activation row into 1x32 BF16 tiles. x_pages = 0: the source is a flat BF16 vector (the layer
//    boundary's normed hidden, fused_decode/boundary.py: value h at byte 2 h), read in one piece; otherwise tile j is
//    row (j / x_pages) of 32x32 BF16 tile page (j % x_pages) of the source tensor (the [heads, head_dim] SDPA output
//    read head by head; a row of a 32x32 tile is 16 values in face 0/2 and 16 in face 1/3). A last 1x32 tile carries
//    `nbias` ones (they multiply the bias rows of each weight column's extra K tile).
// 2. Scatters each 1x32 output tile (output column n = col0 + c, columns >= n_tiles are bank padding and dropped):
//    out_mode 3: BF16 attention heads: column n belongs to Q (n < q_cols), K (n < q_cols + k_cols) or V, head
//                h = n' / head_tiles, half t = n' % head_tiles; written into row h of tile t of that [heads, head_dim]
//                tensor (the decode head-split layout);
//    out_mode 4: router top-k: the single output tile holds the n_logits = k_cols expert logits; the top q_cols = k
//                experts (ties to the lower id) and the softmax over their logits are written as the first k
//                entries of the UINT16 ids (out) and BF16 scores (k_addr) row-major buffers. This core has no FPU:
//                the softmax runs in Q16 fixed point (exp2 = integer shift x degree-5 polynomial, ~1e-5 relative),
//                then rounds to BF16;
//    out_mode 5: flat BF16 output (the boundary's partial-sum input): column n -> one 64-byte write at byte 64 n.
// 3. topk = 1 (the fused decode LM head, fused_decode/terminal.py): also keeps this core's top-32 output values (ties
//    to the lower output index; index = 32 n + lane) and, after the last column, writes them as 32 UINT32 order keys +
//    32 UINT32 indices into slot `list_idx` of the merge core's list buffer and increments its semaphore `merge_sem`.
// 4. QKV_ROPE_KV (out_mode 3, one user; fused_decode/stream.py QKVRopeCache): before the activation arrives, loads the
//    RoPE inputs of the compute kernel (kernels/stream_qkv_rope_compute.cpp: [I, T] and the position's cos / sin
//    rows), reads the position and its paged-cache block (unpaged caches: block 0) and prefetches the cache tile of
//    each of this core's K / V columns (ROWS > 1: the cache tiles are read after the activation). Q columns (rotated
//    by compute) go to the Q heads as above; a K (rotated) or V column is converted to BFP8 into row
//    (position % block) of its cache tile, which is written back. The weight slots are in K / V-first order
//    (qkv_column_order): slot c's output column is runtime arg 12 + c.
//
// 5. ROWS = M > 1 (DFlash verify, M token rows): activation / output tiles are [M, 32] (faces [M, 16]: row r of
//    face f at byte f * 32 M + 32 r). x_pages = 0: the source is the boundary's row-tile layout
//    (fused_decode/boundary.py: [M, 32] tiles back to back, read in one piece); otherwise row r of tile j is row (j /
//    x_pages) of tile page (r * x_pages + j % x_pages) of an [M users, 32, 32 x_pages] source (the SDPA output, one
//    user per token row). out_mode 5 writes tile n to bytes [n * 64 M, (n + 1) * 64 M) of the row-tile partial;
//    out_mode 3 writes row r of column n into the [M users, 32 heads, head_tiles * 32] head tensors (tile page r *
//    head_tiles + t). out_mode 6 (LM head argmax): no logits are written; this core keeps each row's largest output
//    (BF16 order key, ties to the lower output index 32 n + lane) and finally writes [key, index] (2 x UINT32) per row
//    at byte 8 rows * list_idx of `out` (list_idx: runtime arg 5); the cores' / devices' lists are merged on device
//    (dflash/argmax.py ArgmaxGather). LM_TOPK = K: each row's K largest outputs instead, one page per (core, row)
//    (dflash/lm_rescore.py LMRescore reads them).
//
// runtime args: [x_addr, out_addr, col0, k_addr, v_addr, lists_addr, list_idx]; QKV_ROPE_KV: [x_addr, out_addr, col0,
//                k_addr, v_addr, rot, cos, sin, pos, page_table, k_cache, v_cache addresses, slot columns...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// the fused boundary's x-ready count: 1 (one notifier) or the multi-core boundary's helper count (direct notify)
#ifndef BND_WAIT_N
#define BND_WAIT_N 1
#endif
#include "router_topk.hpp"
#include "bfp8_row.hpp"

#ifndef ROWS
#define ROWS 1
#endif

#ifdef QKV_ROPE_KV
// Writes 32 BF16 values (1x32 tile at src: values 0-15 then 16-31) into row rr of the BFP8_b 32x32 tile at tile
// (L1): per 16-value face row the shared exponent is the largest exponent (bfp8_row.hpp). lo / hi: the 16 BF16 values
// of the row's first / second half.
inline void bfp8_write_row(uint32_t tile, uint32_t lo, uint32_t hi, uint32_t rr) {
    for (uint32_t hh = 0; hh < 2; ++hh) {
        uint32_t shared, man[4];
        bfp8_row::half_row(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hh == 0 ? lo : hi), shared, man);
        bfp8_row::put(tile, rr, hh, shared, man);
    }
}
#endif

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t kx = get_compile_time_arg_val(3);
    constexpr uint32_t x_pages = get_compile_time_arg_val(4);
    constexpr uint32_t nbias = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t out_mode = get_compile_time_arg_val(8);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t x_stage_pages = get_compile_time_arg_val(10);  // > 0: source in DRAM, staged in L1 first
    constexpr uint32_t cb_stage = get_compile_time_arg_val(11);
    constexpr uint32_t q_cols = get_compile_time_arg_val(12);
    constexpr uint32_t k_cols = get_compile_time_arg_val(13);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t notify_ct = v_args.next_compile_time_args_offset();
    constexpr uint32_t notify = get_compile_time_arg_val(notify_ct);  // 1: increment the boundary sender's semaphore
    constexpr uint32_t notify_x = get_compile_time_arg_val(notify_ct + 1);
    constexpr uint32_t notify_y = get_compile_time_arg_val(notify_ct + 2);
    constexpr uint32_t notify_sem = get_compile_time_arg_val(notify_ct + 3);
    // 1: x is the normed output of a boundary fused into this op (fused_decode/boundary.py: consumer_parts); wait for
    // the boundary core's notification (program semaphore 0) before reading it.
    constexpr uint32_t wait_x = get_compile_time_arg_val(notify_ct + 4);
    constexpr uint32_t topk = get_compile_time_arg_val(notify_ct + 5);
    constexpr uint32_t merge_x = get_compile_time_arg_val(notify_ct + 6);
    constexpr uint32_t merge_y = get_compile_time_arg_val(notify_ct + 7);
    constexpr uint32_t merge_sem = get_compile_time_arg_val(notify_ct + 8);
#ifdef QKV_ROPE_KV
    // Fused RoPE + KV-cache write (one user): see fused_decode/stream.py QKVRopeCache.
    constexpr uint32_t rk = notify_ct + 9;
    constexpr uint32_t cb_rot = get_compile_time_arg_val(rk);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(rk + 1);
    constexpr uint32_t cb_kv = get_compile_time_arg_val(rk + 2);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(rk + 3);
    constexpr uint32_t block_size = get_compile_time_arg_val(rk + 4);
    constexpr uint32_t kv_tile_bytes = get_compile_time_arg_val(rk + 5);
    constexpr uint32_t paged = get_compile_time_arg_val(rk + 6);  // 0: unpaged caches, the user's rows are block 0
    constexpr auto rot_args = TensorAccessorArgs<rk + 7>();
    constexpr auto cos_args = TensorAccessorArgs<rot_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    const auto s_rot = TensorAccessor(rot_args, get_arg_val<uint32_t>(5), kv_tile_bytes);
    const auto s_cos = TensorAccessor(cos_args, get_arg_val<uint32_t>(6), 2048);
    const auto s_sin = TensorAccessor(sin_args, get_arg_val<uint32_t>(7), 2048);
    const auto s_pos = TensorAccessor(pos_args, get_arg_val<uint32_t>(8), 64);
    const auto s_pt = TensorAccessor(pt_args, get_arg_val<uint32_t>(9), 64);
    const auto s_kc = TensorAccessor(kc_args, get_arg_val<uint32_t>(10), kv_tile_bytes);
    const auto s_vc = TensorAccessor(vc_args, get_arg_val<uint32_t>(11), kv_tile_bytes);
    // Output column of weight slot c (runtime args 12..: the slot order of fused_decode/stream.py qkv_column_order).
    auto slot_column = [](uint32_t c) -> uint32_t { return get_arg_val<uint32_t>(12 + c); };
#endif
    constexpr uint32_t K = 32;
    uint32_t top_key[K];
    uint32_t top_idx[K];
    if constexpr (topk) {
        for (uint32_t i = 0; i < K; ++i) {
            top_key[i] = 0;  // below every real value's key
            top_idx[i] = 0xFFFFFFFFu;
        }
    }

    constexpr uint32_t rows = ROWS;
    static_assert(rows == 1 || topk == 0, "multi-row: no LM-head top-k yet");
    static_assert(out_mode != 4 || rows == 1 || (k_cols == 32 && q_cols <= 8), "router: 32 experts, top-k <= 8");
#if !(defined(MOE_SPLIT_ROWS) && defined(MOE_EARLY_UNION))
    static_assert(out_mode != 4 || rows == 1, "multi-row router: the fused MoE program's (dflash/moe_fused.py)");
#endif
    constexpr uint32_t tiny_bytes = 64 * rows;  // one [rows, 32] BF16 tile
    constexpr uint32_t row_face = 32 * rows;    // one [rows, 16] face
    constexpr uint32_t half_row = 32;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t chunk = 16;

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_out = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto s_k = TensorAccessor(k_args, k_addr, out_page_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, out_page_bytes);

#if defined(QKV_ROPE_KV) && ROWS == 1
    // Before the activation (none of this depends on it, and a fused boundary is still running): RoPE inputs for
    // compute: [I, T] and the position's cos / sin rows (row 0 of the first user's [32, head_dim] shard: 16 values in
    // face 0, 16 in face 1 of each 32-wide tile).
    cb_reserve_back(cb_rot, 2);
    const uint32_t rot_l1 = get_write_ptr(cb_rot);
    noc_async_read(s_rot.get_noc_addr(0), rot_l1, kv_tile_bytes);
    noc_async_read(s_rot.get_noc_addr(1), rot_l1 + kv_tile_bytes, kv_tile_bytes);
    cb_reserve_back(cb_cs, 2 * head_tiles);
    const uint32_t cs_l1 = get_write_ptr(cb_cs);
    for (uint32_t t = 0; t < head_tiles; ++t) {
        noc_async_read(s_cos.get_noc_addr(t), cs_l1 + t * tiny_bytes, half_row);
        noc_async_read(s_cos.get_noc_addr(t) + face_bytes, cs_l1 + t * tiny_bytes + half_row, half_row);
        noc_async_read(s_sin.get_noc_addr(t), cs_l1 + (head_tiles + t) * tiny_bytes, half_row);
        noc_async_read(s_sin.get_noc_addr(t) + face_bytes, cs_l1 + (head_tiles + t) * tiny_bytes + half_row, half_row);
    }
    // The position and its page (64-byte aligned reads, DRAM).
    cb_reserve_back(cb_kv, 1 + cols);
    const uint32_t kv_l1 = get_write_ptr(cb_kv);
    noc_async_read(s_pos.get_noc_addr(0), kv_l1, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1)[0];
    const uint32_t blk_idx = pos / block_size;
    uint32_t blk = 0;
    if constexpr (paged) {
        noc_async_read(s_pt.get_noc_addr(0) + ((blk_idx * 4) & ~63u), kv_l1, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        blk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1)[(blk_idx & 15)];
    }
    const uint32_t r = pos % block_size;
    // Paged cache tile of K / V output column n: row r of block blk, kv head m / head_tiles, tile m % head_tiles.
    auto kv_cache_addr = [&](uint32_t n) -> uint64_t {
        const bool is_k = n < q_cols + k_cols;
        const uint32_t m = is_k ? n - q_cols : n - q_cols - k_cols;
        const uint32_t tile =
            ((blk * num_kv_heads + m / head_tiles) * (block_size / 32) + r / 32) * head_tiles + m % head_tiles;
        return is_k ? s_kc.get_noc_addr(tile) : s_vc.get_noc_addr(tile);
    };
    // Prefetch the cache tile of each of this core's K / V columns (page 1 + c of cb_kv): each output column then
    // only patches its row and writes the tile back, with no DRAM round trip in the output loop.
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = slot_column(c);
        if (n >= q_cols && n < n_tiles) {
            noc_async_read(kv_cache_addr(n), kv_l1 + (1 + c) * kv_tile_bytes, kv_tile_bytes);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_push_back(cb_rot, 2);
    cb_push_back(cb_cs, 2 * head_tiles);
#elif defined(QKV_ROPE_KV)
    // rows token rows at positions pos[0..rows) (consecutive, one user: DFlash verify). cos / sin: row-major tables,
    // one page per position; row r of [rows, 32] tile t gets values 32 t .. 32 t + 31 of row pos[r]. The rows' K / V
    // land in at most two cache tiles per column (group 0: the first row's 32-row tile, group 1: the last row's).
    constexpr uint32_t rope_row = head_tiles * 64;
    const auto s_ctab = TensorAccessor(cos_args, get_arg_val<uint32_t>(6), rope_row);
    const auto s_stab = TensorAccessor(sin_args, get_arg_val<uint32_t>(7), rope_row);
    constexpr uint32_t scr_pages = 3;
    constexpr uint32_t off_cs = 192;  // [pos page | page-table chunk 0 | chunk 1 | cos rows | sin rows]
    static_assert(off_cs + 2 * rows * rope_row <= scr_pages * kv_tile_bytes, "rope scratch");
    static_assert(rows <= 32 && rows <= block_size, "rows span at most two 32-row cache tiles");
    cb_reserve_back(cb_rot, 2);
    const uint32_t rot_l1 = get_write_ptr(cb_rot);
    noc_async_read(s_rot.get_noc_addr(0), rot_l1, kv_tile_bytes);
    noc_async_read(s_rot.get_noc_addr(1), rot_l1 + kv_tile_bytes, kv_tile_bytes);
    cb_reserve_back(cb_kv, scr_pages + 2 * cols);
    const uint32_t kv_l1 = get_write_ptr(cb_kv);
    noc_async_read(s_pos.get_noc_addr(0), kv_l1, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t row_pos[rows];
    for (uint32_t i = 0; i < rows; ++i) {
        row_pos[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1)[i];
    }
    const uint32_t bi0 = row_pos[0] / block_size, bi1 = row_pos[rows - 1] / block_size;
    uint32_t blk0 = 0, blk1 = 0;
    if constexpr (paged) {
        noc_async_read(s_pt.get_noc_addr(0) + ((bi0 * 4) & ~63u), kv_l1 + 64, 64);
        noc_async_read(s_pt.get_noc_addr(0) + ((bi1 * 4) & ~63u), kv_l1 + 128, 64);
    }
    for (uint32_t i = 0; i < rows; ++i) {
        noc_async_read(s_ctab.get_noc_addr(row_pos[i]), kv_l1 + off_cs + i * rope_row, rope_row);
        noc_async_read(s_stab.get_noc_addr(row_pos[i]), kv_l1 + off_cs + (rows + i) * rope_row, rope_row);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    if constexpr (paged) {
        blk0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1 + 64)[bi0 & 15];
        blk1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1 + 128)[bi1 & 15];
    }
    const uint32_t tr0 = (row_pos[0] % block_size) / 32, tr1 = (row_pos[rows - 1] % block_size) / 32;
    const bool two_groups = bi0 != bi1 || tr0 != tr1;
    auto row_group = [&](uint32_t i) -> uint32_t {
        return (row_pos[i] / block_size == bi0 && (row_pos[i] % block_size) / 32 == tr0) ? 0 : 1;
    };
    // cos / sin rows -> [rows, 32] tiles: cos_t at cs page t, sin_t at page head_tiles + t.
    cb_reserve_back(cb_cs, 2 * head_tiles);
    const uint32_t cs_l1 = get_write_ptr(cb_cs);
    for (uint32_t tab = 0; tab < 2; ++tab) {
        for (uint32_t i = 0; i < rows; ++i) {
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1 + off_cs + (tab * rows + i) * rope_row);
            for (uint32_t t = 0; t < head_tiles; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        cs_l1 + (tab * head_tiles + t) * tiny_bytes + f * row_face + i * half_row);
                    for (uint32_t w = 0; w < 8; ++w) {
                        dst[w] = src[t * 16 + f * 8 + w];
                    }
                }
            }
        }
    }
    // Cache tile of K / V output column n in group g.
    auto kv_cache_addr_g = [&](uint32_t n, uint32_t g) -> uint64_t {
        const bool is_k = n < q_cols + k_cols;
        const uint32_t m = is_k ? n - q_cols : n - q_cols - k_cols;
        const uint32_t blk = g == 0 ? blk0 : blk1;
        const uint32_t tr = g == 0 ? tr0 : tr1;
        const uint32_t tile =
            ((blk * num_kv_heads + m / head_tiles) * (block_size / 32) + tr) * head_tiles + m % head_tiles;
        return is_k ? s_kc.get_noc_addr(tile) : s_vc.get_noc_addr(tile);
    };
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_push_back(cb_rot, 2);
    cb_push_back(cb_cs, 2 * head_tiles);
    // this core's K / V cache tiles (read-modify-write of the block rows): issued once x is in (below), they are
    // first needed when the first K / V column comes out
    auto read_kv_tiles = [&]() {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t n = slot_column(c);
            if (n >= q_cols && n < n_tiles) {
                const uint32_t t0 = kv_l1 + (scr_pages + 2 * c) * kv_tile_bytes;
                noc_async_read(kv_cache_addr_g(n, 0), t0, kv_tile_bytes);
                if (two_groups) {
                    noc_async_read(kv_cache_addr_g(n, 1), t0 + kv_tile_bytes, kv_tile_bytes);
                }
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    };
#endif

    cb_reserve_back(cb_x, kx + 1);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    if constexpr (wait_x) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), BND_WAIT_N);
    }
#ifdef LIN_X_WAIT_SEM
    // x is produced in the same program (o_proj after row attention, dflash/verify.py): LIN_X_WAIT_N increments first
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(LIN_X_WAIT_SEM)), LIN_X_WAIT_N);
#endif
    uint32_t x_stage = 0;
    if constexpr (x_stage_pages > 0) {
        // DRAM source: 32-byte reads must be 64-byte aligned there, so stage the whole tiles in L1 first and pick
        // the rows out of the local copy.
        cb_reserve_back(cb_stage, x_stage_pages);
        x_stage = get_write_ptr(cb_stage);
        for (uint32_t p = 0; p < x_stage_pages; ++p) {  // x_stage_pages = rows * x_pages
            noc_async_read(s_x.get_noc_addr(p), x_stage + p * 2048, 2048);
        }
        noc_async_read_barrier();
    }
    constexpr uint32_t gather_pages = x_pages > 0 ? x_pages : 1;
    if constexpr (x_pages == 0) {
        // page by page: x may be sharded over the boundary helpers (fused_decode/boundary.py)
#ifdef BND_PH
        // sharded over the helpers, BND_PH pages each and contiguous in a helper's L1: one read per helper (the
        // sharded accessor's per-page address math made the 23 page reads take ~2 us)
        for (uint32_t off = 0; off < kx * tiny_bytes; off += BND_PH * 2048) {
            const uint32_t n = kx * tiny_bytes - off < BND_PH * 2048 ? kx * tiny_bytes - off : BND_PH * 2048;
            noc_async_read(s_x.get_noc_addr(off / 2048), x_l1 + off, n);
        }
#else
        for (uint32_t p = 0; p * 2048 < kx * tiny_bytes; ++p) {
            const uint32_t n = kx * tiny_bytes - p * 2048 < 2048 ? kx * tiny_bytes - p * 2048 : 2048;
            noc_async_read(s_x.get_noc_addr(p), x_l1 + p * 2048, n);
        }
#endif
    }
#ifdef X_GATHER_SPLIT
    // Split gather (multi-row o_proj): this core gathers activation tiles [j0, j1) only and copies them to the same
    // offset of every other core's cb_x (same CB address on all cores), then counts the n - 1 other cores' copies
    // (program semaphore X_GATHER_SPLIT). runtime args 5..: [core index, n, (noc x, y) x n].
    const uint32_t gs_i = get_arg_val<uint32_t>(5);
    const uint32_t gs_n = get_arg_val<uint32_t>(6);
    const uint32_t gs_per = (kx + gs_n - 1) / gs_n;
    const uint32_t j0 = gs_i * gs_per < kx ? gs_i * gs_per : kx;
    const uint32_t j1 = j0 + gs_per < kx ? j0 + gs_per : kx;
#else
    constexpr uint32_t j0 = 0;
    constexpr uint32_t j1 = kx;
#endif
    for (uint32_t j = j0; x_pages > 0 && j < j1; ++j) {
        const uint32_t row = j / gather_pages;
#ifndef X_SPEC_TT
        const uint32_t off = row < 16 ? row * half_row : 2 * face_bytes + (row - 16) * half_row;  // faces 0/1, 2/3
#endif
        for (uint32_t r = 0; r < rows; ++r) {
#ifdef X_SPEC_TT
            // block attention's packed output (dflash/block_attention.py): row r, head `row` is tile row
            // 8 (r % 4) + row % 8 of tile ((row / 8) * X_SPEC_TT + r / 4) * x_pages + d
            const uint32_t page = ((row / 8) * X_SPEC_TT + r / 4) * gather_pages + j % gather_pages;
            const uint32_t prow = 8 * (r % 4) + row % 8;
            const uint32_t off = prow < 16 ? prow * half_row : 2 * face_bytes + (prow - 16) * half_row;
#else
            const uint32_t page = r * gather_pages + j % gather_pages;
#endif
            const uint64_t src = x_stage_pages > 0 ? get_noc_addr(x_stage + page * 2048) : s_x.get_noc_addr(page);
            noc_async_read(src + off, x_l1 + j * tiny_bytes + r * half_row, half_row);
            noc_async_read(src + off + face_bytes, x_l1 + j * tiny_bytes + row_face + r * half_row, half_row);
        }
    }
    // Bias multipliers: `nbias` ones at the start of every row (face 0), zeros elsewhere.
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x_l1 + kx * tiny_bytes);
    for (uint32_t i = 0; i < 32 * rows; ++i) {
        ones[i] = (i < 16 * rows && i % 16 < nbias) ? 0x3F80 : 0;  // BF16 1.0
    }
    noc_async_read_barrier();
#ifdef X_GATHER_SPLIT
    if (j1 > j0) {
        const uint32_t sem = get_semaphore(X_GATHER_SPLIT);
        for (uint32_t c = 0; c < gs_n; ++c) {
            if (c != gs_i) {
                const uint32_t cx = get_arg_val<uint32_t>(7 + 2 * c), cy = get_arg_val<uint32_t>(8 + 2 * c);
                noc_async_write(
                    x_l1 + j0 * tiny_bytes, get_noc_addr(cx, cy, x_l1 + j0 * tiny_bytes), (j1 - j0) * tiny_bytes);
            }
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < gs_n; ++c) {
            if (c != gs_i) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(7 + 2 * c), get_arg_val<uint32_t>(8 + 2 * c), sem), 1);
            }
        }
        noc_async_atomic_barrier();
    }
    {
        // every other core that owns tiles copied them here
        const uint32_t owners = (kx + gs_per - 1) / gs_per;
        const uint32_t others = owners - (j1 > j0 ? 1 : 0);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(X_GATHER_SPLIT)), others);
    }
#endif
    cb_push_back(cb_x, kx + 1);
#if defined(QKV_ROPE_KV) && ROWS > 1
    read_kv_tiles();
#endif

#ifdef LM_TOPK
    // out_mode 6 with LM_TOPK = K (DFlash two-stage LM head, dflash/lm_rescore.py): each row's K largest outputs,
    // sorted descending (ties: the lower index first); the list is rows x K [key, index] entries
    constexpr uint32_t KT = LM_TOPK;
    int32_t tk[rows][KT];
    uint32_t ti[rows][KT];
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < KT; ++k) {
            tk[r][k] = INT32_MIN;
            ti[r][k] = 0xFFFFFFFFu;
        }
    }
#endif
    uint32_t best_key[rows];
    uint32_t best_idx[rows];
    if constexpr (out_mode == 6) {
        for (uint32_t r = 0; r < rows; ++r) {
            best_key[r] = static_cast<uint32_t>(INT32_MIN);  // below every key (keys are 16-bit signed)
            best_idx[r] = 0xFFFFFFFFu;
        }
    }
    uint32_t exp_chunks = 0;
    if constexpr (out_mode == 4) {
        cb_reserve_back(cb_scr, 1);
        exp_chunks = get_write_ptr(cb_scr);
    }
    for (uint32_t c = 0; c < cols; ++c) {
        cb_wait_front(cb_out, 1);
#ifdef QKV_ROPE_KV
        const uint32_t n = slot_column(c);
#else
        const uint32_t n = col0 + c;
#endif
        if (n < n_tiles) {
            const uint32_t src = get_read_ptr(cb_out);
            if constexpr (out_mode == 4 && rows == 1) {
                router_topk::topk_softmax<q_cols, k_cols>(
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src),
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(exp_chunks));
                noc_async_write(exp_chunks, s_out.get_noc_addr(0), chunk);
                noc_async_write(exp_chunks + chunk, s_k.get_noc_addr(0), chunk);
            } else if constexpr (out_mode == 4) {
                // Multi-row (the fused MoE's router, dflash/moe_fused.py): per row top-k + softmax, then the union of
                // the rows' experts (ascending id): out (UINT16, 64 entries): [U, id_0 .. id_{U-1}, 0 ..]; k_addr
                // (BF16 [rows, 32] row-major): row r, entry i = row r's softmax weight of union expert i (0 when row r
                // did not route to it).
                invalidate_l1_cache();
#if defined(MOE_SPLIT_ROWS) && defined(MOE_EARLY_UNION)
                // MOE_EARLY_UNION: the top-4 ids of every row (rows [rows / 2, rows) on NCRISC, MOE_SPLIT_SEM 1), the
                // union out and the gate|up / down cores released (zeroed counters, semaphore 1) before the softmax;
                // then the weights (NCRISC's rows: MOE_SPLIT_SEM 2) and semaphore MOE_WSEM on the same cores
                {
                    volatile tt_l1_ptr uint32_t* ids =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exp_chunks + 128 + 64 * rows + 128);
                    volatile tt_l1_ptr uint32_t* wq = ids + rows * q_cols;
                    volatile tt_l1_ptr uint32_t* keys = wq + rows * q_cols;
                    volatile tt_l1_ptr uint32_t* split =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MOE_SPLIT_SEM));
                    router_topk::rows_ids<q_cols, rows>(reinterpret_cast<const uint32_t*>(src), 0, rows / 2, ids, keys);
                    noc_semaphore_wait_min(split, 1);
                    invalidate_l1_cache();
                    uint32_t slot[32];
                    const uint32_t u = router_topk::union_ids<q_cols, rows>(
                        ids, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exp_chunks), slot);
                    noc_async_write(exp_chunks, s_out.get_noc_addr(0), 128);
                    volatile tt_l1_ptr uint32_t* ctr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
                    for (uint32_t i = 0; i < 32 * 4; ++i) {
                        ctr[i] = 0;
                    }
                    noc_async_write_barrier();
                    const uint32_t n_rel = get_arg_val<uint32_t>(6);
                    for (uint32_t i = 0; i < n_rel; ++i) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                get_arg_val<uint32_t>(7 + 2 * i), get_arg_val<uint32_t>(8 + 2 * i), get_semaphore(1)),
                            1);
                    }
                    router_topk::rows_softmax<q_cols>(0, rows / 2, keys, wq);
                    noc_semaphore_wait_min(split, 2);
                    invalidate_l1_cache();
                    router_topk::union_weights<q_cols, rows>(
                        u, slot, ids, wq, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exp_chunks + 128));
                    noc_async_write(exp_chunks + 128, s_k.get_noc_addr(0), 64 * rows);
                    noc_async_write_barrier();
                    for (uint32_t i = 0; i < n_rel; ++i) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                get_arg_val<uint32_t>(7 + 2 * i),
                                get_arg_val<uint32_t>(8 + 2 * i),
                                get_semaphore(MOE_WSEM)),
                            1);
                    }
                    noc_async_atomic_barrier();
                }
#endif
            } else if constexpr (out_mode == 6) {
                // BF16 b -> signed order key: s = (int16) b; key = s ^ ((s >> 31) & 0x7FFF) (monotonic in the
                // value, as signed ints). Word loads (two values), values in index order (strict > keeps the lowest).
                invalidate_l1_cache();
                const uint32_t* words = reinterpret_cast<const uint32_t*>(src);
#ifdef LM_TOPK
                for (uint32_t r = 0; r < rows; ++r) {
                    int32_t* k_ = tk[r];
                    uint32_t* i_ = ti[r];
                    int32_t thr = k_[KT - 1];
                    auto ins = [&](int32_t key, uint32_t idx) __attribute__((always_inline)) {
                        uint32_t p = KT - 1;
                        while (p > 0 && key > k_[p - 1]) {
                            k_[p] = k_[p - 1];
                            i_[p] = i_[p - 1];
                            --p;
                        }
                        k_[p] = key;
                        i_[p] = idx;
                        thr = k_[KT - 1];
                    };
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t* w = words + (f * row_face + r * half_row) / 4;
                        const uint32_t base = n * 32 + f * 16;
                        for (uint32_t i = 0; i < 8; ++i) {
                            const uint32_t v = w[i];
                            if (thr >= 0) {
                                // both values' keys <= thr (>= 0) unless one is positive with magnitude bits > thr:
                                // per 16-bit half, ((h | 0x8000) - (thr + 1)) keeps bit 15 iff (h & 0x7FFF) > thr,
                                // and ~h's bit 15 is the sign test (no borrow crosses the halves)
                                const uint32_t k2 = static_cast<uint32_t>(thr + 1) * 0x00010001u;
                                if ((((v | 0x80008000u) - k2) & ~v & 0x80008000u) == 0) {
                                    continue;
                                }
                            }
                            int32_t lo = static_cast<int32_t>(static_cast<int16_t>(v & 0xFFFF));
                            int32_t hi = static_cast<int32_t>(v) >> 16;
                            lo ^= (lo >> 31) & 0x7FFF;
                            hi ^= (hi >> 31) & 0x7FFF;
                            if (lo > thr) {
                                ins(lo, base + 2 * i);
                            }
                            if (hi > thr) {
                                ins(hi, base + 2 * i + 1);
                            }
                        }
                    }
                }
#else
                for (uint32_t r = 0; r < rows; ++r) {
                    int32_t bk = static_cast<int32_t>(best_key[r]);
                    uint32_t bi = best_idx[r];
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t* w = words + (f * row_face + r * half_row) / 4;
                        const uint32_t base = n * 32 + f * 16;
                        for (uint32_t i = 0; i < 8; ++i) {
                            const uint32_t v = w[i];
                            if (bk >= 0) {  // the word test of the LM_TOPK scan above, against the running best
                                const uint32_t k2 = static_cast<uint32_t>(bk + 1) * 0x00010001u;
                                if ((((v | 0x80008000u) - k2) & ~v & 0x80008000u) == 0) {
                                    continue;
                                }
                            }
                            int32_t lo = static_cast<int32_t>(static_cast<int16_t>(v & 0xFFFF));
                            int32_t hi = static_cast<int32_t>(v) >> 16;
                            lo ^= (lo >> 31) & 0x7FFF;
                            hi ^= (hi >> 31) & 0x7FFF;
                            if (lo > bk) {
                                bk = lo;
                                bi = base + 2 * i;
                            }
                            if (hi > bk) {
                                bk = hi;
                                bi = base + 2 * i + 1;
                            }
                        }
                    }
                    best_key[r] = static_cast<uint32_t>(bk);
                    best_idx[r] = bi;
                }
#endif
            } else if constexpr (out_mode == 5) {
                noc_async_write(src, s_out.get_noc_addr(n * tiny_bytes / 2048, (n * tiny_bytes) % 2048), tiny_bytes);
            }
#if defined(QKV_ROPE_KV) && ROWS == 1
            else if (n >= q_cols) {
                // K (rotated) / V row -> row r of its prefetched paged cache tile, written back.
                const uint32_t tile_l1 = kv_l1 + (1 + c) * kv_tile_bytes;
                invalidate_l1_cache();  // src was written by the packer
                bfp8_write_row(tile_l1, src, src + 32, r % 32);
                noc_async_write(tile_l1, kv_cache_addr(n), kv_tile_bytes);
            }
#elif defined(QKV_ROPE_KV)
            else if (n >= q_cols) {
                // K (rotated) / V rows -> their prefetched cache tiles (row pos % 32), written back.
                const uint32_t t0 = kv_l1 + (scr_pages + 2 * c) * kv_tile_bytes;
                invalidate_l1_cache();  // src was written by the packer
#ifdef KV_SPLIT
                // the first output column, when it is a K / V column (the reader's runtime arg 4 is then set): rows
                // [rows / 2, rows) were converted by the reader (kernels/stream_linear_reader.cpp) into
                // KV_SPLIT_CB_STAGE
                const uint32_t i1 = c == 0 ? rows / 2 : rows;
#else
                constexpr uint32_t i1 = rows;
#endif
                for (uint32_t i = 0; i < i1; ++i) {
                    bfp8_write_row(
                        t0 + row_group(i) * kv_tile_bytes,
                        src + i * half_row,
                        src + row_face + i * half_row,
                        row_pos[i] % 32);
                }
#ifdef KV_SPLIT
                if (c == 0) {
                    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(KV_SPLIT_SEM)), 1);
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint32_t* st =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(KV_SPLIT_CB_STAGE));
                    for (uint32_t i = rows / 2; i < rows; ++i) {
                        for (uint32_t hh = 0; hh < 2; ++hh) {
                            volatile tt_l1_ptr uint32_t* e = st + 8 * (2 * i + hh);
                            const uint32_t man[4] = {e[1], e[2], e[3], e[4]};
                            bfp8_row::put(t0 + row_group(i) * kv_tile_bytes, row_pos[i] % 32, hh, e[0], man);
                        }
                    }
                }
#endif
                noc_async_write(t0, kv_cache_addr_g(n, 0), kv_tile_bytes);
                if (two_groups) {
                    noc_async_write(t0 + kv_tile_bytes, kv_cache_addr_g(n, 1), kv_tile_bytes);
                }
            }
#endif
            else {
                const uint32_t m = n < q_cols ? n : (n < q_cols + k_cols ? n - q_cols : n - q_cols - k_cols);
                const uint32_t h = m / head_tiles;
#ifndef Q_SPEC_TT
                const uint32_t off = h < 16 ? h * half_row : 2 * face_bytes + (h - 16) * half_row;
#endif
                for (uint32_t r = 0; r < rows; ++r) {
#ifdef Q_SPEC_TT
                    // Q in block attention's packed layout (dflash/block_attention.py): row r, head h at
                    // tile row 8 (r % 4) + h % 8 of tile ((h / 8) * Q_SPEC_TT + r / 4) * head_tiles + t
                    const uint32_t page = ((h / 8) * Q_SPEC_TT + r / 4) * head_tiles + m % head_tiles;
                    const uint32_t prow = 8 * (r % 4) + h % 8;
                    const uint32_t off = prow < 16 ? prow * half_row : 2 * face_bytes + (prow - 16) * half_row;
#else
                    const uint32_t page = r * head_tiles + m % head_tiles;
#endif
                    const uint64_t dst = n < q_cols            ? s_out.get_noc_addr(page, off)
                                         : n < q_cols + k_cols ? s_k.get_noc_addr(page, off)
                                                               : s_v.get_noc_addr(page, off);
                    noc_async_write(src + r * half_row, dst, half_row);
                    noc_async_write(src + row_face + r * half_row, dst + face_bytes, half_row);
                }
                if constexpr (topk) {
                    // Running top-K of this core's outputs (columns arrive in index order, so a value equal to the
                    // current K-th one has a larger index and does not enter).
                    volatile tt_l1_ptr uint16_t* val = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
                    for (uint32_t j = 0; j < 32; ++j) {
                        const uint32_t b = val[j];
                        const uint32_t key = (b & 0x8000) ? (~b & 0xFFFFu) : (b | 0x8000u);
                        if (key > top_key[K - 1]) {
                            uint32_t i = K - 1;
                            while (i > 0 && top_key[i - 1] < key) {
                                top_key[i] = top_key[i - 1];
                                top_idx[i] = top_idx[i - 1];
                                --i;
                            }
                            top_key[i] = key;
                            top_idx[i] = n * 32 + j;
                        }
                    }
                }
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    if constexpr (out_mode == 6) {
        // Through the (now drained) activation staging of cb_x.
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_l1);
        const uint32_t list_idx = get_arg_val<uint32_t>(5);
#ifdef LM_TOPK
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < KT; ++k) {
                list[2 * (r * KT + k)] = static_cast<uint32_t>(tk[r][k]);
                list[2 * (r * KT + k) + 1] = ti[r][k];
            }
        }
        // one page (out_page_bytes = 8 K) per (core, row): page list_idx * rows + r (spread over the banks)
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write(x_l1 + r * KT * 8, s_out.get_noc_addr(list_idx * rows + r), 8 * KT);
        }
#else
        for (uint32_t r = 0; r < rows; ++r) {
            list[2 * r] = best_key[r];
            list[2 * r + 1] = best_idx[r];
        }
        noc_async_write(x_l1, s_out.get_noc_addr(0, list_idx * 8 * rows), 8 * rows);
#endif
    }
    if constexpr (topk) {
        // The list goes out through the (now drained) weight-gather staging of cb_x: K keys then K indices.
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_l1);
        for (uint32_t i = 0; i < K; ++i) {
            list[i] = top_key[i];
            list[K + i] = top_idx[i];
        }
        const uint32_t lists_addr = get_arg_val<uint32_t>(5);
        const uint32_t list_idx = get_arg_val<uint32_t>(6);
        noc_async_write(x_l1, get_noc_addr(merge_x, merge_y, lists_addr + list_idx * 8 * K), 8 * K);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(merge_sem)), 1);
        noc_async_atomic_barrier();
    }
    noc_async_write_barrier();
#ifdef MOE_FUSED
    // Router of the fused MoE program (dflash/moe_fused.py): the per-slot expert-done counters (runtime arg 5) were
    // zeroed and the gate|up and down cores released (runtime args 6: their count, 7..: noc x, y) after the union,
    // above (MOE_EARLY_UNION).
    {
        const uint32_t n = get_arg_val<uint32_t>(6);
#ifdef MOE_PRED
        // Next round's prediction for this layer: the P union experts routed by the most rows (ties: lower slot),
        // written once every gate|up / down kernel read the current one (program semaphore MOE_PRED_SEM).
        // runtime args 7 + 2 n: this layer's prediction address (this core), 8 + 2 n: the readers to wait for.
        {
            volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(exp_chunks);
            volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(exp_chunks + 128);
            const uint32_t u = uni[0];
            uint32_t cnt[32];
            for (uint32_t i = 0; i < u; ++i) {
                cnt[i] = 0;
                for (uint32_t r = 0; r < rows; ++r) {
                    cnt[i] += wts[r * 32 + i] != 0 ? 1 : 0;
                }
            }
            uint32_t pick[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            for (uint32_t k = 0; k < MOE_PRED && k < u; ++k) {
                uint32_t best = 32;
                for (uint32_t i = 0; i < u; ++i) {
                    bool taken = false;
                    for (uint32_t j = 0; j < k; ++j) {
                        taken |= pick[j] == uni[1 + i];
                    }
                    if (!taken && (best == 32 || cnt[i] > cnt[best])) {
                        best = i;
                    }
                }
                pick[k] = uni[1 + best];
            }
            const uint32_t pred_addr = get_arg_val<uint32_t>(7 + 2 * n);
            noc_semaphore_wait(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MOE_PRED_SEM)),
                get_arg_val<uint32_t>(8 + 2 * n));
            volatile tt_l1_ptr uint32_t* pr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pred_addr);
            for (uint32_t k = 0; k < 8; ++k) {
                pr[k] = pick[k];
            }
        }
#endif
    }
#endif
    if constexpr (notify) {
        // Fused all-reduce send (fused_decode/boundary.py: DecodeBoundary.sending_program): this core's columns of the
        // partial sum are written; tell the boundary core's sender.
        noc_semaphore_inc(get_noc_addr(notify_x, notify_y, get_semaphore(notify_sem)), 1);
        noc_async_atomic_barrier();
    }
#ifdef QKV_NOTIFY_N
    // QKV fused with row attention (dflash/verify.py): Q and this core's K / V cache rows are written; increment
    // program semaphore QKV_NOTIFY_SEM on every row-attention core (runtime args QKV_NOTIFY_ARG..: their noc x, y)
    noc_async_write_barrier();
    for (uint32_t i = 0; i < QKV_NOTIFY_N; ++i) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(QKV_NOTIFY_ARG + 2 * i),
                get_arg_val<uint32_t>(QKV_NOTIFY_ARG + 2 * i + 1),
                get_semaphore(QKV_NOTIFY_SEM)),
            1);
    }
    noc_async_atomic_barrier();
#endif
}

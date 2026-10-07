// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + output scatter (BRISC, NOC1) for the decode streamed linear op (experts/stream.py:
// LinearStream).
//
// 1. Gathers the activation row into 1x32 BF16 tiles. x_pages = 0: the source is a flat BF16 vector (the layer
//    boundary's normed hidden, tt/decode_boundary.py: value h at byte 2 h), read in one piece; otherwise tile j is
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
// 3. topk = 1 (the fused decode LM head, tt/decode_terminal.py): also keeps this core's top-32 output values (ties to
//    the lower output index; index = 32 n + lane) and, after the last column, writes them as 32 UINT32 order keys + 32
//    UINT32 indices into slot `list_idx` of the merge core's list buffer and increments its semaphore `merge_sem`.
//
// runtime args: [x_addr, out_addr, col0, k_addr, v_addr, lists_addr, list_idx]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "router_topk.hpp"

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
    // 1: x is the normed output of a boundary fused into this op (tt/decode_boundary.py: consumer_parts); wait for
    // the boundary core's notification (program semaphore 0) before reading it.
    constexpr uint32_t wait_x = get_compile_time_arg_val(notify_ct + 4);
    constexpr uint32_t topk = get_compile_time_arg_val(notify_ct + 5);
    constexpr uint32_t merge_x = get_compile_time_arg_val(notify_ct + 6);
    constexpr uint32_t merge_y = get_compile_time_arg_val(notify_ct + 7);
    constexpr uint32_t merge_sem = get_compile_time_arg_val(notify_ct + 8);
    constexpr uint32_t K = 32;
    uint32_t top_key[K];
    uint32_t top_idx[K];
    if constexpr (topk) {
        for (uint32_t i = 0; i < K; ++i) {
            top_key[i] = 0;  // below every real value's key
            top_idx[i] = 0xFFFFFFFFu;
        }
    }

    constexpr uint32_t tiny_bytes = 64;
    constexpr uint32_t half_row = 32;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t chunk = 16;

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_out = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto s_k = TensorAccessor(k_args, k_addr, out_page_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, out_page_bytes);

    cb_reserve_back(cb_x, kx + 1);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    if constexpr (wait_x) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    }
    uint32_t x_stage = 0;
    if constexpr (x_stage_pages > 0) {
        // DRAM source: 32-byte reads must be 64-byte aligned there, so stage the whole tiles in L1 first and pick
        // the rows out of the local copy.
        cb_reserve_back(cb_stage, x_stage_pages);
        x_stage = get_write_ptr(cb_stage);
        for (uint32_t p = 0; p < x_stage_pages; ++p) {
            noc_async_read(s_x.get_noc_addr(p), x_stage + p * 2048, 2048);
        }
        noc_async_read_barrier();
    }
    constexpr uint32_t gather_pages = x_pages > 0 ? x_pages : 1;
    if constexpr (x_pages == 0) {
        noc_async_read(s_x.get_noc_addr(0), x_l1, kx * tiny_bytes);
    }
    for (uint32_t j = 0; x_pages > 0 && j < kx; ++j) {
        const uint32_t row = j / gather_pages;
        const uint32_t off = row < 16 ? row * half_row : 2 * face_bytes + (row - 16) * half_row;  // faces 0/1, 2/3
        const uint64_t src =
            x_stage_pages > 0 ? get_noc_addr(x_stage + (j % gather_pages) * 2048) : s_x.get_noc_addr(j % gather_pages);
        noc_async_read(src + off, x_l1 + j * tiny_bytes, half_row);
        noc_async_read(src + off + face_bytes, x_l1 + j * tiny_bytes + half_row, half_row);
    }
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x_l1 + kx * tiny_bytes);
    for (uint32_t i = 0; i < 32; ++i) {
        ones[i] = i < nbias ? 0x3F80 : 0;  // BF16 1.0
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kx + 1);

    uint32_t exp_chunks = 0;
    if constexpr (out_mode == 4) {
        cb_reserve_back(cb_scr, 1);
        exp_chunks = get_write_ptr(cb_scr);
    }
    for (uint32_t c = 0; c < cols; ++c) {
        cb_wait_front(cb_out, 1);
        const uint32_t n = col0 + c;
        if (n < n_tiles) {
            const uint32_t src = get_read_ptr(cb_out);
            if constexpr (out_mode == 4) {
                router_topk::topk_softmax<q_cols, k_cols>(
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src),
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(exp_chunks));
                noc_async_write(exp_chunks, s_out.get_noc_addr(0), chunk);
                noc_async_write(exp_chunks + chunk, s_k.get_noc_addr(0), chunk);
            } else if constexpr (out_mode == 5) {
                noc_async_write(src, s_out.get_noc_addr(n / 32, (n % 32) * tiny_bytes), tiny_bytes);
            } else {
                const uint32_t m = n < q_cols ? n : (n < q_cols + k_cols ? n - q_cols : n - q_cols - k_cols);
                const uint32_t h = m / head_tiles;
                const uint32_t off = h < 16 ? h * half_row : 2 * face_bytes + (h - 16) * half_row;
                const uint64_t dst = n < q_cols            ? s_out.get_noc_addr(m % head_tiles, off)
                                     : n < q_cols + k_cols ? s_k.get_noc_addr(m % head_tiles, off)
                                                           : s_v.get_noc_addr(m % head_tiles, off);
                noc_async_write(src, dst, half_row);
                noc_async_write(src + half_row, dst + face_bytes, half_row);
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
    if constexpr (notify) {
        // Fused all-reduce send (tt/decode_boundary.py: DecodeBoundary.sending_program): this core's columns of the
        // partial sum are written; tell the boundary core's sender.
        noc_semaphore_inc(get_noc_addr(notify_x, notify_y, get_semaphore(notify_sem)), 1);
        noc_async_atomic_barrier();
    }
}

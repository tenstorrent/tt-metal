// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + output scatter (BRISC, NOC1) for the decode streamed linear op (experts/stream.py:
// LinearStream).
//
// 1. Gathers the activation row into 1x32 BF16 tiles: tile j is row (j / x_pages) of 32x32 BF16 tile page
//    (j % x_pages) of the source tensor (any layout TensorAccessor resolves: a width-sharded norm output read
//    row 0 tile by tile, or the [heads, head_dim] SDPA output read head by head). A row of a 32x32 tile is 16
//    values in face 0/2 and 16 in face 1/3. A last 1x32 tile carries `nbias` ones (they multiply the bias rows of
//    each weight column's extra K tile).
// 2. Scatters each 1x32 output tile (output column n = col0 + c, columns >= n_tiles are bank padding and dropped):
//    out_mode 3: BF16 attention heads: column n belongs to Q (n < q_cols), K (n < q_cols + k_cols) or V, head
//                h = n' / head_tiles, half t = n' % head_tiles; written into row h of tile t of that [heads, head_dim]
//                tensor (the decode head-split layout). With k_cols = 0 and head_tiles = W / 32 this is the packed
//                [32, W] all-reduce input (column n -> row n / head_tiles of tile n % head_tiles);
//    out_mode 4: router top-k: the single output tile holds the n_logits = k_cols expert logits; the top q_cols = k
//                experts (ties to the lower id) and the softmax over their logits are written as the first k
//                entries of the UINT16 ids (out) and BF16 scores (k_addr) row-major buffers. This core has no FPU:
//                the softmax runs in Q16 fixed point (exp2 = integer shift x degree-5 polynomial, ~1e-5 relative),
//                then rounds to BF16.
//
// runtime args: [x_addr, out_addr, col0, k_addr, v_addr]

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
    for (uint32_t j = 0; j < kx; ++j) {
        const uint32_t row = j / x_pages;
        const uint32_t off = row < 16 ? row * half_row : 2 * face_bytes + (row - 16) * half_row;  // faces 0/1, 2/3
        const uint64_t src =
            x_stage_pages > 0 ? get_noc_addr(x_stage + (j % x_pages) * 2048) : s_x.get_noc_addr(j % x_pages);
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
            } else {
                const uint32_t m = n < q_cols ? n : (n < q_cols + k_cols ? n - q_cols : n - q_cols - k_cols);
                const uint32_t h = m / head_tiles;
                const uint32_t off = h < 16 ? h * half_row : 2 * face_bytes + (h - 16) * half_row;
                const uint64_t dst = n < q_cols            ? s_out.get_noc_addr(m % head_tiles, off)
                                     : n < q_cols + k_cols ? s_k.get_noc_addr(m % head_tiles, off)
                                                           : s_v.get_noc_addr(m % head_tiles, off);
                noc_async_write(src, dst, half_row);
                noc_async_write(src + half_row, dst + face_bytes, half_row);
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection activation gather + output scatter (BRISC, NOC1) for the routed-expert stream
// (experts/stream.py).
//
// 1. in0 = k segments of seg_tiles 1x32 tiles: segment e is the score-weighted SwiGLU activation of routed expert e
//    (one [I_pad] BF16 row of the compact activation buffer, already scaled by w_e) followed by one tile holding
//    w_e in its first `nbias` entries, which multiply the bias rows of the weight stream (sum_e w_e * b_e).
// 2. Each 1x32 BF16 output tile (one 32-wide column n of the expert sum) is written into the packed BF16
//    all-reduce input ([32, pack_tiles * 32], hidden value h at row h / (32 * pack_tiles)): row n / pack_tiles of
//    32x32 tile n % pack_tiles, as two 32-byte writes (16 values in face 0/2, 16 in face 1/3). Columns past the
//    hidden size (bank padding) are dropped.
//
// runtime args: [act_addr, scores_addr, out_addr, col0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_sel = get_compile_time_arg_val(4);
    constexpr uint32_t nbias = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);  // real output column tiles
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t pack_tiles = get_compile_time_arg_val(9);
    constexpr auto act_args = TensorAccessorArgs<10>();
    constexpr auto scores_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    constexpr uint32_t tiny_bytes = 64;
    constexpr uint32_t half_row = 32;
    constexpr uint32_t face_bytes = 512;

    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_scores = TensorAccessor(scores_args, scores_addr, 64);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_scores.get_noc_addr(0), scr, 64);

    cb_reserve_back(cb_in0, num_sel * seg_tiles);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t e = 0; e < num_sel; ++e) {
        noc_async_read(s_act.get_noc_addr(e), in0 + e * seg_tiles * tiny_bytes, act_row_bytes);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t e = 0; e < num_sel; ++e) {
        volatile tt_l1_ptr uint16_t* aug =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in0 + (e * seg_tiles + seg_tiles - 1) * tiny_bytes);
        const uint16_t w = scores[e];
        for (uint32_t i = 0; i < 32; ++i) {
            aug[i] = i < nbias ? w : 0;
        }
    }
    cb_push_back(cb_in0, num_sel * seg_tiles);

    for (uint32_t c = 0; c < cols; ++c) {
        cb_wait_front(cb_out, 1);
        const uint32_t n = col0 + c;
        if (n < n_tiles) {
            const uint32_t src = get_read_ptr(cb_out);
            const uint32_t r = n / pack_tiles;
            const uint32_t off = r < 16 ? r * half_row : 2 * face_bytes + (r - 16) * half_row;
            const uint64_t dst = s_out.get_noc_addr(n % pack_tiles, off);
            noc_async_write(src, dst, half_row);
            noc_async_write(src + half_row, dst + face_bytes, half_row);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}

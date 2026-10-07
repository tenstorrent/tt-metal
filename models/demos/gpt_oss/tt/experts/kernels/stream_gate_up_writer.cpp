// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + SwiGLU output scatter (BRISC, NOC1) for the routed-expert gate|up stream (experts/stream.py).
//
// 1. Gathers row 0 of the [32, hidden] BF16 tile-layout MoE input (one token) into 1x32 tiles: row 0 of a 32x32
//    tile is the first 16 values of face 0 and of face 1. A last 1x32 tile carries `nbias` ones, which multiply
//    the bias rows stored in the extra K tile of every weight column.
// 2. Reads the k routing scores (BF16) and hands them to the compute MATH thread through the TRISC mailbox
//    (FP32 bits), which scales each expert's SwiGLU output by its score.
// 3. Writes each 1x32 output tile (32 BF16 values) to its place in the compact [k, I_pad] row-major activation
//    buffer the down projection reads (row e, columns 32 j .. 32 j + 31).
//
// runtime args: [x_addr, act_addr, act_tile0, scores_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_tile0 = get_arg_val<uint32_t>(2);
    const uint32_t scores_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t kx = get_compile_time_arg_val(2);
    constexpr uint32_t nbias = get_compile_time_arg_val(3);
    constexpr uint32_t pairs = get_compile_time_arg_val(4);
    constexpr uint32_t num_sel = get_compile_time_arg_val(5);
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t tile_page_bytes = get_compile_time_arg_val(7);  // BF16 32x32 tile page
    constexpr uint32_t cb_scr = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_half_bytes = 16 * 2;  // 16 BF16 values
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t tiny_bytes = 2 * row_half_bytes;

    const auto s_x = TensorAccessor(x_args, x_addr, tile_page_bytes);
    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_scores = TensorAccessor(scores_args, scores_addr, 64);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_scores.get_noc_addr(0), scr, 64);

    cb_reserve_back(cb_x, kx + 1);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kx; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        noc_async_read(src, x_l1 + k * tiny_bytes, row_half_bytes);
        noc_async_read(src + face_bytes, x_l1 + k * tiny_bytes + row_half_bytes, row_half_bytes);
    }
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x_l1 + kx * tiny_bytes);
    for (uint32_t i = 0; i < 32; ++i) {
        ones[i] = i < nbias ? 0x3F80 : 0;  // BF16 1.0
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kx + 1);
    volatile tt_l1_ptr uint16_t* scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t e = 0; e < num_sel; ++e) {
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(scores[e]) << 16);
    }

    for (uint32_t e = 0; e < num_sel; ++e) {
        for (uint32_t p = 0; p < pairs; ++p) {
            cb_wait_front(cb_act, 1);
            const uint32_t src = get_read_ptr(cb_act);
            noc_async_write(src, s_act.get_noc_addr(e, (act_tile0 + p) * tiny_bytes), tiny_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_act, 1);
        }
    }
    noc_async_write_barrier();
}

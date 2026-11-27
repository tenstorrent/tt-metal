// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Streamed QKV matmul + RoPE (TRISC) of the fused decode layer (fused_decode/stream.py LinearStream, out_mode 3 with
// rope): one 1x32 output tile per weight column, y[col] = in0 . W[:, col] over kt K tiles (custom_mm, LoFi, FP32
// accumulation). Q and K columns (n < q_cols + k_cols) are rotated in place, as rotary_embedding_llama does per 32-wide
// tile of a head:
//   out = y * cos_t + (y @ T) * sin_t        (t = the column's tile within its head, T the 32x32 pair-rotation matrix)
// y is packed once (BF16, the precision the separate RoPE op received), then four 1-tile custom_mm calls load
// y @ I, y @ T, cos_t @ I and sin_t @ I into DST 0..3 (I, T: exact in BF8), and SFPU ops on the result rows combine
// them. V columns are packed as computed.
//
// runtime args: the output column of each of this core's weight slots (fused_decode/stream.py qkv_column_order)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

// SFPU ops over the result rows only: a [ROWS, 32] matmul result occupies the first ROWS rows of DST faces 0 and 1
// (one SFPU iteration covers two face rows; ROWS = 1: 2 iterations as for tile heights <= 4).
#ifndef ROWS
#define ROWS 1
#endif
constexpr uint32_t kRowIters = ROWS <= 4 ? 2 : ROWS / 2;
ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}
ALWI void add_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, kRowIters, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t cb_y = get_compile_time_arg_val(5);    // packed y (1x32 BF16)
    constexpr uint32_t cb_rot = get_compile_time_arg_val(6);  // [I, T] BF8 32x32 tiles
    constexpr uint32_t cb_cs =
        get_compile_time_arg_val(7);  // 1x32 BF16 tiles: cos_0 .. cos_{ht-1}, sin_0 .. sin_{ht-1}
    constexpr uint32_t rope_cols = get_compile_time_arg_val(8);  // q_cols + k_cols
    constexpr uint32_t q_cols = get_compile_time_arg_val(9);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(10);

    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_in0, cb_w, cb_out);

    bool rope_inputs = false;
    cb_wait_front(cb_in0, kt);
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = get_arg_val<uint32_t>(c);
        const bool rope = n < rope_cols;
        tile_regs_acquire();
        cb_wait_front(cb_w, kt);
        custom_mm_block<dest_layout>(cb_in0, cb_w, 0, 0, 0, kt);
        cb_pop_front(cb_w, kt);
        tile_regs_commit();
        const uint32_t cb_dst = rope ? cb_y : cb_out;
        cb_reserve_back(cb_dst, 1);
        tile_regs_wait();
        pack_tile(0, cb_dst);
        tile_regs_release();
        cb_push_back(cb_dst, 1);
        if (!rope) {
            continue;
        }
        if (!rope_inputs) {
            cb_wait_front(cb_rot, 2);
            cb_wait_front(cb_cs, 2 * head_tiles);
            rope_inputs = true;
        }
        const uint32_t t = (n < q_cols ? n : n - q_cols) % head_tiles;
        cb_wait_front(cb_y, 1);
        custom_mm_block_init_short<transpose, dest_layout>(cb_y, cb_rot, cb_out);
        tile_regs_acquire();
        custom_mm_block<dest_layout>(cb_y, cb_rot, 0, 0, 0, 1);                // y
        custom_mm_block<dest_layout>(cb_y, cb_rot, 0, 1, 1, 1);                // y @ T
        custom_mm_block<dest_layout>(cb_cs, cb_rot, t, 0, 2, 1);               // cos_t
        custom_mm_block<dest_layout>(cb_cs, cb_rot, head_tiles + t, 0, 3, 1);  // sin_t
        mul_binary_tile_init();
        mul_rows(0, 2, 0);
        mul_rows(1, 3, 1);
        add_binary_tile_init();
        add_rows(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_y, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        custom_mm_block_init_short<transpose, dest_layout>(cb_in0, cb_w, cb_out);
    }
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dest_layout>(cb_out);
}

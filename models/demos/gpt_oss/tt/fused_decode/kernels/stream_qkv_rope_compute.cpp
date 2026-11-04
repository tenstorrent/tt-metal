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
// QKV_ROPE_EXACT (fused_decode/config.py QKV_ROPE_EXACT): the LoFi loads above keep hidden + 6 mantissa bits of y, cos
// and sin (SrcB), so each loses its last BF16 bit toward zero. Instead y, cos and sin are copied into DST exactly
// (unpack A, datacopy), and y @ T is computed exactly as y_hi @ T + y_lo @ T: y_hi = y through the LoFi load (y with
// its last bit dropped, exactly what LoFi keeps), y_lo = y - y_hi (one bit); both are packed after y in cb_y and
// multiply [T, T] (rot tiles 1, 2) in one K = 2 custom_mm.
//
// QKV_ROPE_EXACT_LITE: y, cos and sin are copied into DST exactly as above, but y @ T keeps the one-tile LoFi load (y
// without its last bit; T is exact): no split pass, two SrcA format switches per column instead of four.
//
// runtime args: the output column of each of this core's weight slots (fused_decode/stream.py qkv_column_order)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#if defined(QKV_ROPE_EXACT) || defined(QKV_ROPE_EXACT_LITE)
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

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
#ifdef QKV_ROPE_EXACT
ALWI void sub_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, kRowIters, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}
#endif
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
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);

    bool rope_inputs = false;
    cb_wait_front(cb_in0, kt);
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = get_arg_val<uint32_t>(c);
        const bool rope = n < rope_cols;
        tile_regs_acquire();
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_in0, cb_w, 0, 0, 0, kt);
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
#ifdef QKV_ROPE_EXACT
        constexpr uint32_t rot_tiles = 3;  // [I, T, T]
#else
        constexpr uint32_t rot_tiles = 2;  // [I, T]
#endif
        if (!rope_inputs) {
            cb_wait_front(cb_rot, rot_tiles);
            cb_wait_front(cb_cs, 2 * head_tiles);
            rope_inputs = true;
        }
        const uint32_t t = (n < q_cols ? n : n - q_cols) % head_tiles;
        cb_wait_front(cb_y, 1);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_y, cb_rot, cb_out);
#ifdef QKV_ROPE_EXACT
        // split y = y_hi + y_lo (both exact through LoFi), packed after y in cb_y
        tile_regs_acquire();
        custom_mm_block<true>(cb_y, cb_rot, 0, 0, 0, 1);  // y_hi: y @ I through LoFi
        reconfig_data_format_srca(cb_rot, cb_y);
        copy_init(cb_y);
        copy_tile(cb_y, 0, 1);  // y, exact
        sub_binary_tile_init();
        sub_rows(1, 0, 1);  // y_lo = y - y_hi
        tile_regs_commit();
        cb_reserve_back(cb_y, 2);
        tile_regs_wait();
        pack_tile(0, cb_y);
        pack_tile(1, cb_y);
        tile_regs_release();
        cb_push_back(cb_y, 2);
        cb_wait_front(cb_y, 3);
        reconfig_data_format_srca(cb_y, cb_rot);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_y, cb_rot, cb_out);
        tile_regs_acquire();
        custom_mm_block<true>(cb_y, cb_rot, 1, 1, 1, 2);  // y_hi @ T + y_lo @ T = y @ T, exact
        reconfig_data_format_srca(cb_rot, cb_y);
        copy_init(cb_y);
        copy_tile(cb_y, 0, 0);                // y
        copy_tile(cb_cs, t, 2);               // cos_t
        copy_tile(cb_cs, head_tiles + t, 3);  // sin_t
        reconfig_data_format_srca(cb_y, cb_rot);
#elif defined(QKV_ROPE_EXACT_LITE)
        tile_regs_acquire();
        custom_mm_block<true>(cb_y, cb_rot, 0, 1, 1, 1);  // y @ T (LoFi: y without its last bit)
        reconfig_data_format_srca(cb_rot, cb_y);
        copy_init(cb_y);
        copy_tile(cb_y, 0, 0);                // y, exact
        copy_tile(cb_cs, t, 2);               // cos_t, exact
        copy_tile(cb_cs, head_tiles + t, 3);  // sin_t, exact
        reconfig_data_format_srca(cb_y, cb_rot);
#else
        tile_regs_acquire();
        custom_mm_block<true>(cb_y, cb_rot, 0, 0, 0, 1);                // y
        custom_mm_block<true>(cb_y, cb_rot, 0, 1, 1, 1);                // y @ T
        custom_mm_block<true>(cb_cs, cb_rot, t, 0, 2, 1);               // cos_t
        custom_mm_block<true>(cb_cs, cb_rot, head_tiles + t, 0, 3, 1);  // sin_t
#endif
        mul_binary_tile_init();
        mul_rows(0, 2, 0);
        mul_rows(1, 3, 1);
        add_binary_tile_init();
        add_rows(0, 1, 0);
        tile_regs_commit();
#ifdef QKV_ROPE_EXACT
        cb_pop_front(cb_y, 3);
#else
        cb_pop_front(cb_y, 1);
#endif
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);
    }
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dense_packing>();
}

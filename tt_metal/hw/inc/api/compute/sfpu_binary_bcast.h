// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#if defined(TRISC_MATH) && (defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE))
#include "llk_math_eltwise_binary_sfpu_binary_bcast.h"
#endif

namespace ckernel {

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)

// ============================================================================
// BCAST_COL: broadcast column 0 of the bcast tile across all 32 tile columns
// ============================================================================

// clang-format off
/**
 * Initialize SFPU state for {add,sub,mul}-with-column-broadcast. Must be
 * called once (before any acquire_dst / sfpu_*_bcast_col pair) to configure
 * the persistent LREG_MASK and replay-buffer slot used by the broadcast
 * helper. A single init is shared across ADD / SUB / MUL because only the
 * broadcast state (which depends on BCAST_DIM) is persisted; the binop
 * opcode is selected per call.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_bcast_col_init() { MATH((llk_math_eltwise_binary_sfpu_bcast_col_init())); }

ALWI void sfpu_sub_bcast_col_init() { MATH((llk_math_eltwise_binary_sfpu_sub_bcast_col_init())); }

ALWI void sfpu_add_bcast_col_init() { MATH((llk_math_eltwise_binary_sfpu_add_bcast_col_init())); }

ALWI void sfpu_mul_bcast_col_init() { MATH((llk_math_eltwise_binary_sfpu_mul_bcast_col_init())); }

// clang-format off
/**
 * Apply a binop between `dst_data_idx` and the column-broadcast of column 0
 * of `dst_col_vec_idx`, entirely within DST registers. Result is written
 * in-place into `dst_data_idx`:
 *
 *     sub:  DST[dst_data_idx][r][c] -= DST[dst_col_vec_idx][r][0]
 *     add:  DST[dst_data_idx][r][c] += DST[dst_col_vec_idx][r][0]
 *     mul:  DST[dst_data_idx][r][c] *= DST[dst_col_vec_idx][r][0]
 *
 * for c in [0..31].
 *
 * Both source registers must contain FP32 data (compile with
 * fp32_dest_acc_en = true). The column-vector register is expected to carry
 * meaningful values only in column 0 of each row (e.g. a row-reduce max /
 * sum output); values in columns 1..31 are ignored. Output is FP32.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Requires a prior call to the matching `sfpu_*_bcast_col_init` (or the
 * shared `sfpu_bcast_col_init`).
 *
 * Return value: None
 *
 * | Argument         | Description                                                         | Type     | Valid Range                                           | Required |
 * |------------------|---------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx     | The index of the data tile in DST (result is written back in-place) | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_col_vec_idx  | The index of the col-vector tile in DST (column 0 is broadcast)     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_sub_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_sub_bcast_col(dst_data_idx, dst_col_vec_idx)));
}

ALWI void sfpu_add_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_add_bcast_col(dst_data_idx, dst_col_vec_idx)));
}

ALWI void sfpu_mul_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_mul_bcast_col(dst_data_idx, dst_col_vec_idx)));
}

// ============================================================================
// BCAST_ROW: broadcast row 0 of the bcast tile across all 32 tile rows
// ============================================================================

// clang-format off
/**
 * Initialize SFPU state for {add,sub,mul}-with-row-broadcast. Must be called
 * once (before any acquire_dst / sfpu_*_bcast_row pair). As with the column
 * variant a single init is shared across ADD / SUB / MUL.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_bcast_row_init() { MATH((llk_math_eltwise_binary_sfpu_bcast_row_init())); }

ALWI void sfpu_sub_bcast_row_init() { MATH((llk_math_eltwise_binary_sfpu_sub_bcast_row_init())); }

ALWI void sfpu_add_bcast_row_init() { MATH((llk_math_eltwise_binary_sfpu_add_bcast_row_init())); }

ALWI void sfpu_mul_bcast_row_init() { MATH((llk_math_eltwise_binary_sfpu_mul_bcast_row_init())); }

// clang-format off
/**
 * Apply a binop between `dst_data_idx` and the row-broadcast of row 0 of
 * `dst_row_vec_idx`, entirely within DST registers. Result is written
 * in-place into `dst_data_idx`:
 *
 *     sub:  DST[dst_data_idx][r][c] -= DST[dst_row_vec_idx][0][c]
 *     add:  DST[dst_data_idx][r][c] += DST[dst_row_vec_idx][0][c]
 *     mul:  DST[dst_data_idx][r][c] *= DST[dst_row_vec_idx][0][c]
 *
 * for r in [0..31].
 *
 * Both source registers must contain FP32 data (compile with
 * fp32_dest_acc_en = true). The row-vector register is expected to carry
 * meaningful values only in row 0 (e.g. layernorm gamma/beta); values in
 * rows 1..31 are ignored. Output is FP32.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Requires a prior call to the matching `sfpu_*_bcast_row_init` (or the
 * shared `sfpu_bcast_row_init`).
 *
 * Return value: None
 *
 * | Argument         | Description                                                         | Type     | Valid Range                                           | Required |
 * |------------------|---------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx     | The index of the data tile in DST (result is written back in-place) | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_row_vec_idx  | The index of the row-vector tile in DST (row 0 is broadcast)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_sub_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_sub_bcast_row(dst_data_idx, dst_row_vec_idx)));
}

ALWI void sfpu_add_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_add_bcast_row(dst_data_idx, dst_row_vec_idx)));
}

ALWI void sfpu_mul_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((llk_math_eltwise_binary_sfpu_mul_bcast_row(dst_data_idx, dst_row_vec_idx)));
}

#endif  // ARCH_WORMHOLE || ARCH_BLACKHOLE

}  // namespace ckernel

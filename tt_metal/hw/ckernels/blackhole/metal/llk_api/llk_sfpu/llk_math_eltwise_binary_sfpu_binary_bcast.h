// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binary_bcast.h"
#include "llk_math_eltwise_binary_sfpu_init.h"

namespace ckernel {

// ---------------------------------------------------------------------------
// Generic in-DST binary-with-broadcast (BCAST_COL / BCAST_ROW).
//
// The raw SFPU kernel iterates over the whole 32x32 tile internally (all 4
// faces) and handles data/bcast/out at arbitrary DST tile indices, so we do
// NOT route through the per-face _llk_math_eltwise_binary_sfpu_params_ loop.
// Instead we open the SFPU section once and invoke the full-tile helper.
// ---------------------------------------------------------------------------

template <ckernel::sfpu::SfpuBcastDim BCAST_DIM>
inline void llk_math_eltwise_binary_sfpu_binary_bcast_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_sfpu_binary_bcast_init_<BCAST_DIM>);
}

template <ckernel::BinaryOp BINOP, ckernel::sfpu::SfpuBcastDim BCAST_DIM>
inline void llk_math_eltwise_binary_sfpu_binary_bcast(
    uint32_t dst_index_data, uint32_t dst_index_bcast, uint32_t dst_index_out) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0);
    ckernel::sfpu::_calculate_sfpu_binary_bcast_full_tile_<BINOP, BCAST_DIM>(
        dst_index_data, dst_index_bcast, dst_index_out);
    _llk_math_eltwise_binary_sfpu_done_();
}

// ---------------------------------------------------------------------------
// Named convenience wrappers for ADD / SUB / MUL with column- or row-
// broadcast (primary use case: SDPA softmax row-max subtraction / row-scale
// for BCAST_COL; layernorm-style row-broadcast for BCAST_ROW). The init
// is shared across binops because it only depends on BCAST_DIM. The result
// is written in-place into `dst_index_data`.
// ---------------------------------------------------------------------------

// --- BCAST_COL ---------------------------------------------------------------

inline void llk_math_eltwise_binary_sfpu_bcast_col_init() {
    llk_math_eltwise_binary_sfpu_binary_bcast_init<ckernel::sfpu::SfpuBcastDim::BCAST_COL>();
}

inline void llk_math_eltwise_binary_sfpu_sub_bcast_col_init() { llk_math_eltwise_binary_sfpu_bcast_col_init(); }

inline void llk_math_eltwise_binary_sfpu_add_bcast_col_init() { llk_math_eltwise_binary_sfpu_bcast_col_init(); }

inline void llk_math_eltwise_binary_sfpu_mul_bcast_col_init() { llk_math_eltwise_binary_sfpu_bcast_col_init(); }

inline void llk_math_eltwise_binary_sfpu_sub_bcast_col(uint32_t dst_index_data, uint32_t dst_index_col_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::SUB, ckernel::sfpu::SfpuBcastDim::BCAST_COL>(
        dst_index_data, dst_index_col_vec, /*dst_index_out=*/dst_index_data);
}

inline void llk_math_eltwise_binary_sfpu_add_bcast_col(uint32_t dst_index_data, uint32_t dst_index_col_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::ADD, ckernel::sfpu::SfpuBcastDim::BCAST_COL>(
        dst_index_data, dst_index_col_vec, /*dst_index_out=*/dst_index_data);
}

inline void llk_math_eltwise_binary_sfpu_mul_bcast_col(uint32_t dst_index_data, uint32_t dst_index_col_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::MUL, ckernel::sfpu::SfpuBcastDim::BCAST_COL>(
        dst_index_data, dst_index_col_vec, /*dst_index_out=*/dst_index_data);
}

// --- BCAST_ROW ---------------------------------------------------------------

inline void llk_math_eltwise_binary_sfpu_bcast_row_init() {
    llk_math_eltwise_binary_sfpu_binary_bcast_init<ckernel::sfpu::SfpuBcastDim::BCAST_ROW>();
}

inline void llk_math_eltwise_binary_sfpu_sub_bcast_row_init() { llk_math_eltwise_binary_sfpu_bcast_row_init(); }

inline void llk_math_eltwise_binary_sfpu_add_bcast_row_init() { llk_math_eltwise_binary_sfpu_bcast_row_init(); }

inline void llk_math_eltwise_binary_sfpu_mul_bcast_row_init() { llk_math_eltwise_binary_sfpu_bcast_row_init(); }

inline void llk_math_eltwise_binary_sfpu_sub_bcast_row(uint32_t dst_index_data, uint32_t dst_index_row_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::SUB, ckernel::sfpu::SfpuBcastDim::BCAST_ROW>(
        dst_index_data, dst_index_row_vec, /*dst_index_out=*/dst_index_data);
}

inline void llk_math_eltwise_binary_sfpu_add_bcast_row(uint32_t dst_index_data, uint32_t dst_index_row_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::ADD, ckernel::sfpu::SfpuBcastDim::BCAST_ROW>(
        dst_index_data, dst_index_row_vec, /*dst_index_out=*/dst_index_data);
}

inline void llk_math_eltwise_binary_sfpu_mul_bcast_row(uint32_t dst_index_data, uint32_t dst_index_row_vec) {
    llk_math_eltwise_binary_sfpu_binary_bcast<ckernel::BinaryOp::MUL, ckernel::sfpu::SfpuBcastDim::BCAST_ROW>(
        dst_index_data, dst_index_row_vec, /*dst_index_out=*/dst_index_data);
}

}  // namespace ckernel

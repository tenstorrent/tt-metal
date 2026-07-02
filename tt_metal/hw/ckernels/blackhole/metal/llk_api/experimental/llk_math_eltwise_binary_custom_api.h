// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_assert.h"
#include "llk_math_common_api.h"
#include "experimental/llk_math_eltwise_binary_custom.h"

/*************************************************************************
 * LLK MATH ELTWISE BINARY CUSTOM - blocked bcast-col paths
 *   SUB: SDPA   |   MUL: indexer_score gate reduction
 *************************************************************************/

/**
 * @brief Init the math (FPU) thread for the SDPA blocked bcast-col SUB path.
 *
 * @tparam math_fidelity: FPU fidelity phase, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param operandA: CB id of srcA; its num_faces drives the addr-mod setup.
 * @param operandB: CB id of srcB (the bcast-col operand).
 * @note Run before @ref llk_math_eltwise_binary_sub_bcast_cols_custom on this thread.
 */
template <MathFidelity math_fidelity>
inline void llk_math_eltwise_binary_sub_bcast_cols_init_custom(
    const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_custom_<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(num_faces);
}

/**
 * @brief SDPA blocked bcast-col SUB over ct_dim column tiles starting at dst_index.
 *
 * @tparam is_fp32_dest_acc_en: Whether dest accumulates in fp32 (selects the dest tile budget).
 * @param dst_index: First destination tile index.
 * @param ct_dim: Number of column tiles written, into dest range [dst_index, dst_index + ct_dim).
 * @note Run @ref llk_math_eltwise_binary_sub_bcast_cols_init_custom first.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sub_bcast_cols_custom(const std::uint32_t dst_index, const std::uint32_t ct_dim = 1) {
    LLK_ASSERT(
        (dst_index + ct_dim <= get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "dst range out of bounds");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    _llk_math_sub_bcast_cols_reuse_custom_(ct_dim);
    math::clear_dst_reg_addr();
}

/**
 * @brief Init the math (FPU) thread for the indexer_score blocked bcast-col MUL path.
 *
 * @param operandA: CB id of srcA; its num_faces drives the addr-mod setup.
 * @param operandB: CB id of srcB (the bcast-col operand).
 * @note No fidelity template arg (unlike the SUB init): the COL bcast init doesn't take one. Run before
 *       @ref llk_math_eltwise_binary_mul_bcast_cols_custom on this thread.
 */
inline void llk_math_eltwise_binary_mul_bcast_cols_init_custom(
    const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_custom_<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(num_faces);
}

/**
 * @brief indexer_score blocked bcast-col MUL with dest-MAC head reduction over ct_dim column tiles.
 *
 * Column j MACs onto dest[dst_index + j], so calling once per head into the same dst_index reduces
 * heads in place (see @ref _llk_math_bcast_cols_reuse_custom_).
 *
 * @param dst_index: First destination tile index.
 * @param ct_dim: Number of column tiles written, into dest range [dst_index, dst_index + ct_dim).
 * @note Run @ref llk_math_eltwise_binary_mul_bcast_cols_init_custom first.
 */
inline void llk_math_eltwise_binary_mul_bcast_cols_custom(
    const std::uint32_t dst_index, const std::uint32_t ct_dim = 1) {
    LLK_ASSERT(
        (dst_index + ct_dim <= get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "dst range out of bounds");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    _llk_math_bcast_cols_reuse_custom_<EltwiseBinaryType::ELWMUL>(ct_dim);
    math::clear_dst_reg_addr();
}

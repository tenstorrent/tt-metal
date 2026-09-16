// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_common_api.h"
#include "experimental/llk_math_eltwise_binary_custom.h"

/*************************************************************************
 * LLK MATH ELTWISE BINARY CUSTOM - SDPA blocked bcast-col SUB (Quasar)
 *************************************************************************/

/**
 * @brief Init the math (FPU) thread for the SDPA blocked bcast-col SUB path.
 *
 * Configures the ALU data-format state from the operand formats (defensively; the compute API also
 * calls state_configure) and validates the tile shape. There is no MOP on this path: the execute call
 * programs the COL reuse addr-mods and emits its ELWSUB stream directly.
 *
 * @tparam math_fidelity: Accepted for API parity; SUB is LoFi-only on Quasar, so the value is unused.
 * @param operandA: DFB id of srcA; its format feeds the ALU format state and its tile shape is
 *        validated (32x32 or 16x32 tiles).
 * @param operandB: DFB id of srcB (the bcast-col operand); its format feeds the ALU format state.
 * @note Run before @ref llk_math_eltwise_binary_sub_bcast_cols_custom on this thread.
 */
template <ckernel::MathFidelity math_fidelity>
inline void llk_math_eltwise_binary_sub_bcast_cols_init_custom(
    const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operandA_id));
    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operandB_id));

    _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
        srcA_format, srcB_format);
    _llk_math_eltwise_binary_init_custom_<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(tensor_shape);
}

/**
 * @brief SDPA blocked bcast-col SUB over ct_dim column tiles starting at dst_index.
 *
 * @tparam is_fp32_dest_acc_en: Accepted for API parity with Blackhole; dest capacity is
 *         taken from the DST_ACCUM_MODE (matching the Blackhole wrapper).
 * @param operandA: DFB id of srcA; its tile shape is derived for the LLK call.
 * @param dst_index: First destination tile index; ct_dim tiles land in [dst_index, dst_index + ct_dim).
 * @param ct_dim: Number of column tiles written.
 * @note Run @ref llk_math_eltwise_binary_sub_bcast_cols_init_custom first.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sub_bcast_cols_custom(
    const std::uint32_t operandA, const std::uint32_t dst_index, const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    // Dest capacity in slots of this tile's height. get_dest_max_tiles keys off the dest footprint in
    // rows, so a 2-face 16x32 tile takes the DstTileShape whose footprint is also two faces
    // (Tile32x16, 32 rows) even though its geometry is transposed: twice as many slots fit as at
    // 32x32, which is the 32-row slot stride the LLK programs from the same shape.
    const std::uint32_t max_dest_tiles =
        tensor_shape.total_num_faces() == ckernel::MAX_NUM_FACES
            ? get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()
            : get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x16>();
    LLK_ASSERT(dst_index + ct_dim <= max_dest_tiles, "dst range out of bounds");

    _llk_math_sub_bcast_cols_reuse_custom_(ct_dim, tensor_shape, dst_index);
}

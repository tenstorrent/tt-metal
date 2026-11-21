// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

/**
 *
 * @brief Initialize FPU to perform a reduce operation
 *
 * @tparam pool_type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam EN_32BIT_DEST: Set to true to use 32bit destination register mode
 * @tparam math_fidelity: Only works for AVG/SUM pool types,  0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls
 * precision of multiplication
 * @param operandA: The input operand Data Flow Buffer identifier
 * @param operandB: The scaler input operand Data Flow Buffer identifier
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim, const bool EN_32BIT_DEST, ckernel::MathFidelity math_fidelity>
inline void llk_math_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    constexpr bool swap_operands = reduce_swaps_operands<pool_type, reduce_dim>();
    // Only column SUM and AVG with MxFp4 data and scaler run on MxFp4_2x_B, like matmul.
    const bool src_2x = pool_type != PoolType::MAX && reduce_dim == ReduceDim::REDUCE_COL &&
                        get_operand_src_format(operandA_id) == static_cast<std::uint32_t>(DataFormat::MxFp4) &&
                        get_operand_src_format(operandB_id) == static_cast<std::uint32_t>(DataFormat::MxFp4);
    // MxFp4_2x_B deviates from unpack_dst_format[]; the tracker keys on formats, so it is still programmed.
    const DataFormat data_format =
        src_2x ? DataFormat::MxFp4_2x_B : static_cast<DataFormat>(unpack_dst_format[operandA_id]);
    const DataFormat scaler_format =
        src_2x ? DataFormat::MxFp4_2x_B : static_cast<DataFormat>(unpack_dst_format[operandB_id]);
    const DataFormat srcA_format = swap_operands ? scaler_format : data_format;
    const DataFormat srcB_format = swap_operands ? data_format : scaler_format;

    // Declare the Dest-derived width for operands copied back from Dest; init still takes the unpacked formats.
    const auto alu_format = [](const DataFormat format) {
        return reduce_moves_dest_to_src<pool_type, reduce_dim>() ? ckernel::math::dest_src_format<EN_32BIT_DEST>(format)
                                                                 : format;
    };

    _configure_alu_cfg_<AluCfgType::DEFAULT, EN_32BIT_DEST>(alu_format(srcA_format), alu_format(srcB_format));
    _llk_math_reduce_init_<pool_type, reduce_dim, EN_32BIT_DEST, math_fidelity>(srcA_format, srcB_format, tensor_shape);
}

/**
 * @brief Math-side reduce uninit. Empty, as on WH/BH where _llk_math_reduce_uninit_ is also empty.
 *
 * Nothing to undo: every FPU math init programs its own SrcA/SrcB formats, so the MxFp4 -> MxFp4_2x_B
 * deviation @ref llk_math_reduce_init applies cannot be observed by the next op.
 *
 * @note The unpacker half is not a no-op -- see @ref llk_unpack_AB_reduce_uninit. The packer edge
 *       mask is cleared separately by llk_pack_reduce_mask_clear.
 */
inline void llk_math_reduce_uninit([[maybe_unused]] const std::uint32_t operandA = 0) {}

/**
 * @brief Perform a reduce operation
 *
 * @tparam type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tile shape determining face count and dest stride.
 */
template <PoolType type, ReduceDim dim>
inline void llk_math_reduce(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape) {
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "llk_math_reduce: dst_index exceeds dest capacity");
    _llk_math_reduce_<type, dim>(dst_index, tensor_shape);
}

template <PoolType type, ReduceDim dim>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    llk_math_reduce<type, dim>(dst_index, tensor_shape);
}

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
 * @tparam is_int_fpu_en: When true for REDUCE_ROW, skip MOP programming (runtime int FPU path)
 * @param operandA: The input operand Data Flow Buffer identifier
 * @param operandB: The scaler input operand Data Flow Buffer identifier
 *
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    const bool EN_32BIT_DEST,
    ckernel::MathFidelity math_fidelity,
    bool is_int_fpu_en = false>
inline void llk_math_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    // Column reduce is a GAPOOL (op-mmul family) and consumes MxFp4 SrcA as the 2x-packed
    // src-register format, exactly like matmul. Derive the effective SrcA format from the L1/src
    // format; the matching unpacker OUT_DATA_FORMAT override lives in llk_unpack_AB_reduce_init.
    // Only REDUCE_COL supports 2x (row/scalar reduce post-pool ELWADDDI does not), and only GAPOOL
    // (SUM/AVG) accepts a 2x SrcA - GMPOOL (MAX) does not.
    const bool srcA_2x = (pool_type == PoolType::SUM || pool_type == PoolType::AVG) &&
                         (reduce_dim == ReduceDim::REDUCE_COL) &&
                         (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4);
    const DataFormat srcA_format =
        srcA_2x ? DataFormat::MxFp4_2x_B : static_cast<DataFormat>(unpack_dst_format[operandA_id]);
    const DataFormat srcB_format = static_cast<DataFormat>(unpack_dst_format[operandB_id]);

    // When srcA_2x, srcA_format deviates from the op-agnostic unpack_dst_format[] table. The tracker
    // keys on the formats, so the deviation is programmed and a repeat call is free.
    _configure_alu_cfg_<AluCfgType::DEFAULT, EN_32BIT_DEST>(srcA_format, srcB_format);
    _llk_math_reduce_init_<pool_type, reduce_dim, EN_32BIT_DEST, math_fidelity, is_int_fpu_en>(
        srcA_format, srcB_format, tensor_shape);
}

/**
 * @brief Math-side reduce uninit. Empty, as on WH/BH where _llk_math_reduce_uninit_ is also empty.
 *
 * Nothing to undo: every math init programs its own SrcA/SrcB formats, so the MxFp4 -> MxFp4_2x_B
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
 * @tparam is_int_fpu_en: When true for REDUCE_ROW, runs the runtime int FPU path instead of the MOP
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tile shape determining face count and dest stride. int FPU path requires default 32x32.
 */
template <PoolType type, ReduceDim dim, bool is_int_fpu_en = false>
inline void llk_math_reduce(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape) {
    if constexpr (is_int_fpu_en) {
        LLK_ASSERT(
            tensor_shape.face_r_dim == DEFAULT_TENSOR_SHAPE.face_r_dim &&
                tensor_shape.face_c_dim == DEFAULT_TENSOR_SHAPE.face_c_dim &&
                tensor_shape.num_faces_r_dim == DEFAULT_TENSOR_SHAPE.num_faces_r_dim &&
                tensor_shape.num_faces_c_dim == DEFAULT_TENSOR_SHAPE.num_faces_c_dim,
            "Int reduce: only default 32x32 tensor_shape supported");
    }
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "llk_math_reduce: dst_index exceeds dest capacity");
    _llk_math_reduce_<type, dim, is_int_fpu_en>(dst_index, tensor_shape);
}

template <PoolType type, ReduceDim dim, bool is_int_fpu_en = false>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    llk_math_reduce<type, dim, is_int_fpu_en>(dst_index, tensor_shape);
}

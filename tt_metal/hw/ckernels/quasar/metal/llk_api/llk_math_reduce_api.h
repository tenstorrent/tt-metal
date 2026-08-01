// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
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
    const DataFormat srcA_format = static_cast<DataFormat>(unpack_dst_format[operandA_id]);
    const DataFormat srcB_format = static_cast<DataFormat>(unpack_dst_format[operandB_id]);

    _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, EN_32BIT_DEST>(srcA_format, srcB_format);
    _llk_math_reduce_init_<pool_type, reduce_dim, math_fidelity, is_int_fpu_en>(tensor_shape);
}

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
inline void llk_math_reduce(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape) {
    if constexpr (is_int_fpu_en) {
        LLK_ASSERT(
            tensor_shape.face_r_dim == DEFAULT_TENSOR_SHAPE.face_r_dim &&
                tensor_shape.face_c_dim == DEFAULT_TENSOR_SHAPE.face_c_dim &&
                tensor_shape.num_faces_r_dim == DEFAULT_TENSOR_SHAPE.num_faces_r_dim &&
                tensor_shape.num_faces_c_dim == DEFAULT_TENSOR_SHAPE.num_faces_c_dim,
            "Int reduce: only default 32x32 tensor_shape supported");
    }
    _llk_math_reduce_<type, dim, is_int_fpu_en>(dst_index, tensor_shape);
}

template <PoolType type, ReduceDim dim, bool is_int_fpu_en = false>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    llk_math_reduce<type, dim, is_int_fpu_en>(dst_index, tensor_shape);
}

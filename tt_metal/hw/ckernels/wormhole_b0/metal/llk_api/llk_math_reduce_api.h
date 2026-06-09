// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

/**
 * @brief Perform a reduction on the math thread, pooling faces into the destination register.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @tparam enforce_fp32_accumulation: Force FP32 accumulation through the transpose (requires is_fp32_dest_acc_en).
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tensor shape describing tile dimensions.
 * @note Call @ref llk_math_reduce_init with matching template args before this function, and
 *       @ref llk_math_reduce_uninit after it to restore modified state.
 */
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce(const uint dst_index, const ckernel::TensorShape& tensor_shape) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, tensor_shape);
}

/**
 * @brief Perform a reduction on the math thread, deriving the tile shape from operand A's circular buffer.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @tparam enforce_fp32_accumulation: Force FP32 accumulation through the transpose (requires is_fp32_dest_acc_en).
 * @param operandA: Circular-buffer index used to derive the tile shape (both operands share face count).
 * @param operandB: Unused.
 * @param dst_index: Tile index into the destination register.
 * @note Call @ref llk_math_reduce_init with matching template args before this function, and
 *       @ref llk_math_reduce_uninit after it to restore modified state.
 */
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operandA);  // both operands must have same number of faces
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, tensor_shape);
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam enforce_fp32_accumulation: Force FP32 accumulation through the transpose (requires is_fp32_dest_acc_en).
 * @note On the unpack thread, pair with @ref llk_unpack_reduce_init (single operand) or
 *       @ref llk_unpack_AB_reduce_init (with scaler operand).
 * @ref llk_math_reduce runs the configured reduction with matching template args.
 */
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce_init() {
    _llk_math_reduce_init_<type, dim, is_fp32_dest_acc_en, math_fidelity, enforce_fp32_accumulation>();
}

/**
 * @brief Uninitialize after a reduce operation, restoring modified state using the operand's data format.
 *
 * @tparam enforce_fp32_accumulation: Whether FP32 accumulation through the transpose was enforced.
 * @param operandA: Circular-buffer index whose source data format is used during restore.
 * @note Reverses @ref llk_math_reduce_init.
 */
template <bool enforce_fp32_accumulation = false>
inline void llk_math_reduce_uninit(const std::uint32_t operandA) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    _llk_math_reduce_uninit_<enforce_fp32_accumulation>(unpack_src_format[operand_id]);
}

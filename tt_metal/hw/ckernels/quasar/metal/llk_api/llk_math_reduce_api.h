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
    // Column reduce is a GAPOOL (op-mmul family) and consumes MxFp4 SrcA as the 2x-packed
    // src-register format, exactly like matmul. Derive the effective SrcA format from the L1/src
    // format; the matching unpacker OUT_DATA_FORMAT override lives in llk_unpack_AB_reduce_init.
    // Only REDUCE_COL supports 2x (row/scalar reduce post-pool ELWADDDI does not).
    const bool srcA_2x = (reduce_dim == ReduceDim::REDUCE_COL) &&
                         (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4);
    const DataFormat srcA_format =
        srcA_2x ? DataFormat::MxFp4_2x_B : static_cast<DataFormat>(unpack_dst_format[operandA_id]);
    const DataFormat srcB_format = static_cast<DataFormat>(unpack_dst_format[operandB_id]);

    // When srcA_2x, srcA_format deviates from the op-agnostic unpack_dst_format[] table that kernel
    // startup (llk_math_hw_configure) already programmed the ALU from and latched as
    // DataFormatConfigSet::DEFAULT. That latch keys on which config set is active, not on the
    // formats, so _configure_default_alu_data_format_state_ would early-return and leave the ALU
    // decoding 2x-packed SrcA as Float16_b. Program the ALU directly in that case; this is still the
    // DEFAULT config shape (implied math format off, no dest-format override), so the latched
    // DataFormatConfigSet::DEFAULT stays truthful and transpose-dest's restore contract holds.
    if (srcA_2x) {
        const bool en_int32_dest_format = _is_src_fmt_int32_dest_compatible_(srcA_format) &&
                                          _is_src_fmt_int32_dest_compatible_(srcB_format) && EN_32BIT_DEST;
        _configure_alu_formats_<false /* EN_IMPLIED_MATH_FORMAT */, EN_32BIT_DEST>(
            srcA_format, srcB_format, en_int32_dest_format, DataFormat::Invalid /* no dest-format override */);
    } else {
        _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, EN_32BIT_DEST>(
            srcA_format, srcB_format);
    }
    _llk_math_reduce_init_<pool_type, reduce_dim, math_fidelity, is_int_fpu_en>(tensor_shape);
}

/**
 * @brief Restore the ALU SrcA/SrcB formats after a column reduce that consumed an MxFp4 (2x) operand.
 *
 * @param operandA: The srcA (data) operand circular buffer (same as reduce init)
 *
 * Undoes the MxFp4 -> MxFp4_2x_B deviation that @ref llk_math_reduce_init programmed into the ALU
 * format registers (column reduce only), restoring the op-agnostic unpack_dst_format[] values so a
 * following op decodes its src registers correctly. Gates on MxFp4 (only column reduce deviated;
 * others are a no-op). Uses _configure_alu_formats_ directly because the still-DEFAULT latch makes
 * _configure_default_alu_data_format_state_ early-return. Pair with @ref llk_unpack_AB_reduce_uninit.
 */
inline void llk_math_reduce_uninit(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    if (static_cast<DataFormat>(get_operand_src_format(operandA_id)) != DataFormat::MxFp4) {
        return;
    }
    // All reduce operands (MxFp4 data + scaler) map to Float16_b in unpack_dst_format[], so
    // reprogramming both ALU srcs to operandA's table value is the correct restore.
    const DataFormat table_fmt = static_cast<DataFormat>(unpack_dst_format[operandA_id]);
    const bool en_int32_dest_format = _is_src_fmt_int32_dest_compatible_(table_fmt) && DST_ACCUM_MODE;
    _configure_alu_formats_<false /* EN_IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
        table_fmt, table_fmt, en_int32_dest_format, DataFormat::Invalid /* no dest-format override */);
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

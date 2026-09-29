// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

/**
*
* @brief Initialize matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA

* @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when
* math is in Fp32 format
* @param operandA: Logical dataflow buffer identifier for input 0 (-> SrcB)
* @param operandB: Logical dataflow buffer identifier for input 1 (-> SrcA)
* @param ct_dim: number of tiles in the column dimension for a matrix multiply
* @param rt_dim: number of tiles in the row dimension for a matrix multiply
*
* This function initializes the matrix multiply operation where:
* Input 0 * Input 1 -> SrcB * SrcA
* Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]
* Output is a matrix block of dimension [rt_dim, ct_dim]
* Operand tile shapes are read from circular-buffer metadata. The 2x path requires full 32x32 tiles.
*/
template <ckernel::MathFidelity math_fidelity>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    // MxFp4 fed to matmul is ALWAYS unpacked as the 2x-packed src-register format (MxFp4_2x_B) on
    // Quasar — there is no non-2x MxFp4 matmul. The generated unpack_dst_format[] table keeps the
    // op-agnostic MX default (Float16_b), so derive the effective src-register format from the L1
    // (src) format here; the matching unpacker OUT_DATA_FORMAT override lives in
    // llk_unpack_AB_matmul_init. This reaches the same HW state the old host-side remap produced.
    const auto matmul_src_reg_format = [](const std::uint32_t op_id) -> DataFormat {
        return (static_cast<DataFormat>(get_operand_src_format(op_id)) == DataFormat::MxFp4)
                   ? DataFormat::MxFp4_2x_B
                   : static_cast<DataFormat>(get_operand_dst_format(op_id));
    };
    const DataFormat srcB_format = matmul_src_reg_format(operandA_id);
    const DataFormat srcA_format = matmul_src_reg_format(operandB_id);
    const ckernel::TensorShape src_b_shape = get_operand_tensor_shape(operandA_id);
    const ckernel::TensorShape src_a_shape = get_operand_tensor_shape(operandB_id);
    LLK_ASSERT(
        is_2x_format(srcA_format) == is_2x_format(srcB_format),
        "SrcA and SrcB must both be 2x formats or both non-2x formats");

    // srcA/srcB above are the per-op effective src-register formats, which for MxFp4 differ from the
    // op-agnostic unpack_dst_format[] table that kernel startup (llk_math_hw_configure) already
    // programmed the ALU from and latched as DataFormatConfigSet::DEFAULT. That latch keys on which
    // config set is active, not on the formats, so _configure_default_alu_data_format_state_ would
    // early-return and leave the ALU decoding 2x-packed src registers as Float16_b while the EN_X2
    // MOP below ran over them. When the formats deviate, program the ALU directly instead; this is
    // still the DEFAULT config shape (implied math format off, no dest-format override), so the
    // latched DataFormatConfigSet::DEFAULT stays truthful and transpose-dest's restore contract holds.
    if ((srcA_format != static_cast<DataFormat>(get_operand_dst_format(operandB_id))) ||
        (srcB_format != static_cast<DataFormat>(get_operand_dst_format(operandA_id)))) {
        const bool en_int32_dest_format = _is_src_fmt_int32_dest_compatible_(srcA_format) &&
                                          _is_src_fmt_int32_dest_compatible_(srcB_format) && DST_ACCUM_MODE;
        _configure_alu_formats_<false /* EN_IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
            srcA_format, srcB_format, en_int32_dest_format, DataFormat::Invalid /* no dest-format override */);
    } else {
        _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
            srcA_format, srcB_format);
    }
    const bool src_2x = is_2x_format(srcA_format) && is_2x_format(srcB_format);
    if (src_2x) {
        _llk_math_matmul_init_<math_fidelity, false /*EN_DI*/, true /*EN_X2*/>(
            ct_dim, rt_dim, src_b_shape, src_a_shape);
    } else {
        LLK_ASSERT(
            ckernel::validate_matmul_tensor_shapes_(src_b_shape, src_a_shape),
            "unsupported SrcB/input0 and SrcA/input1 TensorShape pair for matmul");
        _llk_math_matmul_init_<math_fidelity, false /*EN_DI*/, false /*EN_X2*/>(
            ct_dim, rt_dim, src_b_shape, src_a_shape);
    }
}

/**
 * @brief Restore the ALU SrcA/SrcB formats after a matmul that consumed MxFp4 (2x) operands.
 *
 * @param operandA: The input0 operand circular buffer (matches the matmul init call)
 * @param operandB: The input1 operand circular buffer
 *
 * Undoes the MxFp4 -> MxFp4_2x_B deviation that @ref llk_math_matmul_init programmed into the ALU
 * format registers, restoring the op-agnostic unpack_dst_format[] values so a following non-matmul
 * op decodes its src registers correctly. Only acts when init deviated (an operand was MxFp4). Uses
 * _configure_alu_formats_ directly for the same reason init does: the DataFormatConfigSet::DEFAULT
 * latch (still truthful) makes _configure_default_alu_data_format_state_ early-return.
 *
 * @note Pair with @ref llk_unpack_AB_matmul_uninit (via mm_uninit); call before the next op when an
 * MxFp4 matmul operand is reused by a non-matmul op in the same kernel.
 */
inline void llk_math_matmul_uninit(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const bool deviated =
        (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) ||
        (static_cast<DataFormat>(get_operand_src_format(operandB_id)) == DataFormat::MxFp4);
    if (!deviated) {
        return;
    }
    // Same operand->src mapping as init: In0(operandA)->SrcB, In1(operandB)->SrcA. The table values
    // (unpack_dst_format[]) are the op-agnostic formats (Float16_b for MxFp4).
    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operandA_id));
    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operandB_id));
    const bool en_int32_dest_format = _is_src_fmt_int32_dest_compatible_(srcA_format) &&
                                      _is_src_fmt_int32_dest_compatible_(srcB_format) && DST_ACCUM_MODE;
    _configure_alu_formats_<false /* EN_IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
        srcA_format, srcB_format, en_int32_dest_format, DataFormat::Invalid /* no dest-format override */);
}

/**
 * @brief Performs matrix multiply operation, where Input 0, Input 1 and Output are each 1 tile
 *
 * @param dst_index: Tile index into the destination register
 *
 * This function performs the matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA,
 * Input 0 = 1 tile -> SrcB reg, Input 1 = 1 tile -> SrcA reg,
 * Output = 1 tile -> Dst reg at specified dst_index
 */
inline void llk_math_matmul_tile(const std::uint32_t dst_index) { _llk_math_matmul_tile_(dst_index); }

/**
 *
 * @brief Performs matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA, where
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 *
 * @param ct_dim: number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: number of tiles in the row dimension for a matrix multiply
 *
 * This function performs the matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA,
 * Input 0 dim = [rt_dim, 1] -> SrcB reg, Input 1 dim = [1, ct_dim] -> SrcA reg
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 * This function does not iterate over kt_dim, must iterate over kt_dim externally to this function
 * Dest index is always assumed to start at 0 for this operation
 *
 */
inline void llk_math_matmul_block(const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    _llk_math_matmul_block_(ct_dim, rt_dim);
}

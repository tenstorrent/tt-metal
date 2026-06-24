// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"
#include "llk_operands.h"

/**
 * @brief Performs an in-place 32x32 transpose on a tile in the destination
 * register at dst_index.
 *
 * @tparam transpose_of_faces Transpose faces as well
 * @tparam EN_32BIT_DEST True if dest is in 32-bit mode.
 */
template <bool transpose_of_faces = true, bool EN_32BIT_DEST = false>
inline void llk_math_transpose_dest(uint dst_index) {
    _llk_math_transpose_dest_(dst_index);
}

/**
 * @brief Initializes transpose-dest. For EN_32BIT_DEST, configures the ALU into the FP32-dest
 * mov-ops state that transpose-dest requires (implied math format disabled, en_int32_dest_format
 * off because MOVD2A/B do not honor INT8 math), programming the srcA/srcB ALU format registers
 * from the operand's dest data format, then loads the bank0 replay buffer with the transpose-dest
 * MOP.
 *
 * @note Uses the state-tracked _configure_mov_ops_explicit_alu_data_format_state_ so a following
 * op (datacopy/matmul/reduce/binary) detects the non-default ALU state and reconfigures back to
 * default. The raw _llk_math_upk_to_dest_hw_configure_ path did not update that tracking, leaving
 * stale ALU config that subsequent ops silently skipped reconfiguring.
 *
 * @tparam transpose_of_faces Transpose faces as well.
 * @tparam EN_32BIT_DEST True if dest is in 32-bit mode.
 * @param operand Logical dataflow buffer id whose dest data format programs the transpose ALU config.
 */
template <bool transpose_of_faces = true, bool EN_32BIT_DEST = false>
inline void llk_math_transpose_dest_init([[maybe_unused]] const std::uint32_t operand) {
    if constexpr (EN_32BIT_DEST) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const DataFormat operand_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
        _configure_mov_ops_explicit_alu_data_format_state_<true /*EN_32BIT_DEST*/>(operand_format, operand_format);
    }
    _llk_math_transpose_dest_init_<transpose_of_faces, EN_32BIT_DEST>();
}

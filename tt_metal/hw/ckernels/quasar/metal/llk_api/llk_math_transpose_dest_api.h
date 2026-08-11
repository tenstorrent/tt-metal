// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"
#include "llk_operands.h"

/**
 * @brief Initializes transpose-dest. Configures the ALU state that transpose-dest
 * requires for both 16/32-bit dest (implied math format disabled, en_int32_dest_format off
 * because MOVD2A/B do not honor INT8 math), programming the srcA/srcB ALU format registers from the
 * operand's dest data format and enabling 32-bit dest when EN_32BIT_DEST is set, then loads the
 * bank0 replay buffer with the transpose-dest MOP.
 *
 * @note Uses the state-tracked _configure_mov_ops_explicit_alu_data_format_state_ so a following
 * op (datacopy/matmul/reduce/binary) detects the non-default ALU state and reconfigures back to
 * default. The raw _llk_math_upk_to_dest_hw_configure_ path does not update that tracking, leaving
 * stale ALU config that subsequent ops silently skipped reconfiguring.
 *
 * @tparam transpose_of_faces Transpose faces as well.
 * @tparam EN_32BIT_DEST True if dest is in 32-bit mode.
 * @param operand Logical dataflow buffer id whose dest data format programs the transpose ALU config.
 */
template <bool transpose_of_faces = true, bool EN_32BIT_DEST = false>
inline void llk_math_transpose_dest_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const DataFormat operand_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    _configure_mov_ops_explicit_alu_data_format_state_<EN_32BIT_DEST>(operand_format, operand_format);
    _llk_math_transpose_dest_init_<transpose_of_faces, EN_32BIT_DEST>();
}

/**
 * @brief Performs an in-place 32x32 transpose on a tile in the destination
 * register at dst_index.
 *
 * @param dst_index Index of the tile in the dest register to transpose in place.
 * @note Call @ref llk_math_transpose_dest_init first; it applies the faces / 32-bit configuration,
 * so this runs the pre-configured MOP and takes no template params.
 */
inline void llk_math_transpose_dest(const std::uint32_t dst_index) { _llk_math_transpose_dest_(dst_index); }

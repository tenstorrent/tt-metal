// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_AB_reduce_custom.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/*************************************************************************
 * LLK UNPACK AB REDUCE CUSTOM - Specialized reduce_max_row operations
 *************************************************************************/

/**
 * Initializes unpacker configuration for block-based reduce_max_row operations.
 * Sets up tile dimensions and saves unpacker state that will be modified during operation.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native llk_unpack_AB_reduce_init LLK.
 * Use the standard llk_unpack_AB_reduce_init<ReduceDim::REDUCE_ROW> for general-purpose reduction.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_reduce_block_max_row_init() {
    _llk_unpack_AB_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>();
}

/**
 * Performs unpacking for block-based reduce_max_row operation across multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native llk_unpack_AB LLK.
 * Use the standard llk_unpack_AB<BroadcastType::NONE> in a loop for general-purpose operations.
 */
template <uint32_t block_ct_dim>
inline void llk_unpack_AB_reduce_block_max_row(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t row_start_index) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * row_start_index;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    _llk_unpack_AB_reduce_block_max_row_(address_a, base_address_b);
}

/**
 * Uninitializes block-based reduce_max_row unpacker operation.
 * Restores the unpacker state that was saved during initialization.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native llk_unpack_AB_reduce_init LLK.
 * Use standard LLK cleanup procedures for general-purpose operations.
 */
inline void llk_unpack_AB_reduce_block_max_row_uninit() {
    _llk_unpack_AB_reduce_block_max_row_uninit_(FACE_R_DIM, FACE_R_DIM);
}

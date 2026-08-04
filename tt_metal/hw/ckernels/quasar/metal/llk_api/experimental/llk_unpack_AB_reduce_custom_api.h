// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_AB_reduce_runtime_custom.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE CUSTOM - Specialized reduce_max_row unpack (Quasar)
 *
 * Compile-time-block_ct_dim entry points for the arch-agnostic Compute API (reduce_custom.h) / SDPA.
 *
 * WHY THEY FORWARD TO THE RUNTIME LIB: Quasar has NO separate compile-time LLK lib (no `lltt`;
 * `ckernel_template` takes runtime loop bounds), so these wrappers pass the template block_ct_dim as a
 * runtime argument to the single runtime lib. The template exists purely for Compute-API call-shape
 * parity.
 *
 * Quasar bakes the SrcA buffer descriptor into the unpack MOP, and it is only known once the operand is
 * bound (at execute). block_ct_dim is a template here, so it is available at execute -- so the MOP is
 * programmed at EXECUTE, and init only enables the UNPACKER0 transpose. This keeps the init call
 * operand-free and portable with WH/BH.
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row unpacker (compile-time block_ct_dim). Enables transpose only.
 *
 * @tparam block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode.
 * @tparam respect_trigger  SDPA MOP-split handshake -- unsupported on Quasar; must stay false.
 * @param tensor_shape  Operand tile shape (accepted for WH/BH signature parity; the MOP that uses it is
 *                      programmed at execute).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row_init(const ckernel::TensorShape& tensor_shape) {
    static_assert(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");
    static_assert(!respect_trigger, "respect_trigger (MOP-split handshake) not supported on Quasar");
    (void)tensor_shape;  // the MOP (which uses the shape + buffer descriptor) is programmed at execute
    _llk_unpack_AB_reduce_block_max_row_cfg_(true);
}

/**
 * @brief Programs the block reduce_max_row unpack MOP and unpacks a block of SrcA tiles + one scaler.
 *
 * Resolves operandA/operandB to buffer descriptors + L1 tile indices, then programs the MOP for this
 * block (Quasar needs the SrcA buffer descriptor, only known here) and fires it.
 *
 * @tparam block_ct_dim     Number of tiles in the width dimension processed as one block.
 * @tparam respect_trigger  Unsupported on Quasar; must stay false.
 * @param operandA          SrcA operand (block of tiles) circular buffer identifier.
 * @param operandB          SrcB scaler operand circular buffer identifier.
 * @param row_start_index   Tile offset of the block's first SrcA tile within operandA's CB.
 * @param tensor_shape      Operand tile shape -- the SAME shape the math thread uses.
 */
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t row_start_index,
    const ckernel::TensorShape& tensor_shape) {
    static_assert(!respect_trigger, "respect_trigger (MOP-split handshake) not supported on Quasar");
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(
        block_ct_dim, operandA_id, operandB_id, tensor_shape, respect_trigger);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);
    const std::uint32_t l1_tile_index_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + row_start_index;
    const std::uint32_t l1_tile_index_b = local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx;

    WAYPOINT("URBW");
    _llk_unpack_AB_reduce_block_max_row_runtime_(
        block_ct_dim,
        l1_tile_index_a,
        l1_tile_index_b,
        operandB_id,
        tensor_shape,
        respect_trigger,
        false /*overlap_first_half*/);
    WAYPOINT("URBD");
}

template <bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row_uninit() {
    static_assert(!respect_trigger, "respect_trigger (MOP-split handshake) not supported on Quasar");
    _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(respect_trigger, false /*overlap_first_half*/);
}

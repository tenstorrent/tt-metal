// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

/**
 *
 * @brief Initialize unpack for unpack reduce operations, which unpacks one tile for srcA and one face for srcB
 *
 * @tparam pool_type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param operandA: The srcA operand DFB identifier
 * @param operandB: The srcB operand DFB identifier
 *
 * This function initializes the UNPACKER0 to unpack a single tile from the input DFB to srcA
 * and UNPACKER1 to unpack a single face from the input DFB to srcB, with specified reduce dimension.
 * For REDUCE_ROW SUM/AVG the operands are swapped (see @ref reduce_swaps_operands): operandA (data) goes
 * UNPACKER1 -> SRCB and operandB (scaler) goes UNPACKER0 -> SRCA.
 *
 * Each operand gets a BFD id allocated from the unpack partition (operandA on Unp0 / UNPACR0,
 * operandB on Unp1 / UNPACR1) and its table entry is programmed here; the DFB ids are used only
 * to fetch buffer info, never as BFD ids. One init burns 2 unpack-partition ids, so the partition
 * wraps sooner under mixed workloads — the standard wrap contract (re-init before re-execute)
 * applies.
 *
 * @note A column SUM/AVG with both operands MxFp4 unpacks them as MxFp4_2x_B; undo it with
 * @ref llk_unpack_AB_reduce_uninit.
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    const std::uint8_t bfd_a = llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operandA_id);
    const std::uint8_t bfd_b = llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp1>(operandB_id);

    // Only column SUM and AVG with MxFp4 data and scaler run on MxFp4_2x_B, like matmul.
    const bool src_2x = pool_type != PoolType::MAX && reduce_dim == ReduceDim::REDUCE_COL &&
                        get_operand_src_format(operandA_id) == static_cast<std::uint32_t>(DataFormat::MxFp4) &&
                        get_operand_src_format(operandB_id) == static_cast<std::uint32_t>(DataFormat::MxFp4);

    _llk_unpack_reduce_init_<pool_type, reduce_dim>(
        bfd_a, bfd_b, tensor_shape, ckernel::trisc::NUM_TILES, src_2x /*clear_data_src*/);

    // EN_32BIT_DEST does not affect the MxFp4 -> MxFp4_2x_B reconfig validity today; DST_ACCUM_MODE keeps it honest.
    if (src_2x) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, DST_ACCUM_MODE>(
            get_operand_src_format(operandA_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, DST_ACCUM_MODE>(
            get_operand_src_format(operandB_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
    }
}

/**
 * @brief Undo the MxFp4_2x_B unpacker formats set by @ref llk_unpack_AB_reduce_init.
 *
 * @param operandA: The srcA (data) operand circular buffer (same as reduce init)
 *
 * Both reduce operands of an MxFp4 reduce use operandA's table format, so both unpackers are restored from it.
 * Pair with @ref llk_math_reduce_uninit.
 */
inline void llk_unpack_AB_reduce_uninit(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    if (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, DST_ACCUM_MODE>(
            get_operand_src_format(operandA_id), unpack_dst_format[operandA_id]);
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, DST_ACCUM_MODE>(
            get_operand_src_format(operandA_id), unpack_dst_format[operandA_id]);
    }
}

/**
 *
 * @brief Unpacks binary operands to SrcA & SrcB for reduce kernels
 *
 * @tparam pool_type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 * @param tile_index_a: The L1 index in the input DFB to read from, tile_index_a -> UNPACKER0 -> SRCA
 * @param tile_index_b: The L1 index in the input DFB to read from, tile_index_b -> UNPACKER1 -> SRCB
 *        (for REDUCE_ROW SUM/AVG the operands are swapped, see @ref reduce_swaps_operands:
 *        tile_index_a -> UNPACKER1 -> SRCB, tile_index_b -> UNPACKER0 -> SRCA)
 *
 * This function performs unpacking for reduce kernels, the UNPACKER0 unpacks a single tile from the input DFB
 * to srcA and UNPACKER1 unpacks a single face from the input DFB to srcB.
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    LLK_TDMA_GUARD_NOTE_TDMA(operandA);  // TEN-4746: real unpack (UNPACR) disarms these dfbs
    LLK_TDMA_GUARD_NOTE_TDMA(operandB);
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        ckernel::trisc::BfdResource::Unp0,
        operandA,
        "unpack_AB_reduce operandA DFB differs from the one llk_unpack_AB_reduce_init programmed");
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        ckernel::trisc::BfdResource::Unp1,
        operandB,
        "unpack_AB_reduce operandB DFB differs from the one llk_unpack_AB_reduce_init programmed");

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);

    const std::uint32_t l1_tile_index_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].cursor_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_index_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].cursor_entry_idx + tile_index_b;

    WAYPOINT("UABW");
    _llk_unpack_reduce_<pool_type, reduce_dim>(l1_tile_index_a, l1_tile_index_b, tensor_shape);
    WAYPOINT("UABD");
}

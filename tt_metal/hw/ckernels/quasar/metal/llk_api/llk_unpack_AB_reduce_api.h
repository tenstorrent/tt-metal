// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"
#include "llk_reduce_common_api.h"

namespace ckernel::trisc {
// The scaler DFB whose unpacker (UNP_B) llk_unpack_AB_reduce_init moved to MxFp4_2x_B for a 2x column
// reduce, stored as operand id + 1 so that 0 (REDUCE_2X_SCALER_NONE) means none: the firmware definition
// stays zero-initialized in .tbss. llk_unpack_AB_reduce_uninit is only given the data operand, so it
// restores UNP_B from this. thread_local because each Neo has its own UNP_B and runs its own
// init/uninit: a shared global would let one Neo's uninit clear another's record (tt-llk#1678).
// Defined in tt_metal/hw/firmware/src/tt-2xx/trisc.cc.
extern thread_local std::uint32_t reduce_2x_scaler_operand_plus_one;
}  // namespace ckernel::trisc

constexpr std::uint32_t REDUCE_2X_SCALER_NONE = 0;

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
 *
 * Each operand gets a BFD id allocated from the unpack partition (operandA on Unp0 / UNPACR0,
 * operandB on Unp1 / UNPACR1) and its table entry is programmed here; the DFB ids are used only
 * to fetch buffer info, never as BFD ids. One init burns 2 unpack-partition ids, so the partition
 * wraps sooner under mixed workloads — the standard wrap contract (re-init before re-execute)
 * applies.
 *
 * @note A SUM/AVG column reduce with both operands MxFp4 (see @ref is_2x_column_reduce) runs on the 2x-packed
 * src-register format: init moves both unpackers' OUT_DATA_FORMAT to MxFp4_2x_B and records the scaler so
 * @ref llk_unpack_AB_reduce_uninit can restore it. With an MxFp4 data operand and any other scaler format, the
 * reduce runs on the op-agnostic unpack_dst_format[] formats (Float16_b) instead.
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    const std::uint8_t bfd_a = llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operandA_id);
    const std::uint8_t bfd_b = llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp1>(operandB_id);

    _llk_unpack_reduce_init_<pool_type, reduce_dim>(bfd_a, bfd_b, tensor_shape);

    // A 2x column reduce (both operands MxFp4; see is_2x_column_reduce, shared with llk_math_reduce_init so the
    // unpacker and ALU formats agree) runs both src registers on the 2x-packed format. Override only the unpacker
    // gasket OUT_DATA_FORMAT to MxFp4_2x_B (shadow register; unpacker idle at init before the first UNPACR; buffer
    // descriptor keyed on the MxFp4 L1 format is unchanged): operandA -> SrcA -> UNP_A, operandB -> SrcB -> UNP_B.
    // EN_32BIT_DEST does not affect the MxFp4->MxFp4_2x_B reconfig validity, so pass false.
    if (is_2x_column_reduce<pool_type, reduce_dim>(operandA_id, operandB_id)) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, false /*EN_32BIT_DEST*/>(
            get_operand_src_format(operandA_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, false /*EN_32BIT_DEST*/>(
            get_operand_src_format(operandB_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
        ckernel::trisc::reduce_2x_scaler_operand_plus_one = operandB_id + 1;
    }
}

/**
 * @brief Undo the MxFp4 -> MxFp4_2x_B unpacker OUT_DATA_FORMAT override from @ref llk_unpack_AB_reduce_init.
 *
 * @param operandA: The srcA (data) operand circular buffer (same as reduce init)
 *
 * Restores SrcA's unpacker OUT_DATA_FORMAT to the op-agnostic unpack_dst_format[] value (Float16_b),
 * so a following NON-reduce op on the same MxFp4 buffer unpacks correctly. Needed because non-reduce
 * unpack inits never reprogram OUT_DATA_FORMAT and reconfig_data_format is silently skipped for a
 * same-format operand. Only a 2x column reduce overrode it (SrcA -> UNP_A); restoring an operand that
 * was never overridden just reprograms it to the same table value (harmless), so this gates on MxFp4
 * only and needs no reduce_dim template. The scaler's UNP_B override is restored from the operand
 * this Neo's init recorded (independent of operandA), since this is only given the data operand. Pair
 * with the ALU restore in @ref llk_math_reduce_uninit.
 */
inline void llk_unpack_AB_reduce_uninit(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    if (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, false /*EN_32BIT_DEST*/>(
            get_operand_src_format(operandA_id), unpack_dst_format[operandA_id]);
    }
    if (ckernel::trisc::reduce_2x_scaler_operand_plus_one != REDUCE_2X_SCALER_NONE) {
        const std::uint32_t operandB_id = ckernel::trisc::reduce_2x_scaler_operand_plus_one - 1;
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, false /*EN_32BIT_DEST*/>(
            get_operand_src_format(operandB_id), unpack_dst_format[operandB_id]);
        ckernel::trisc::reduce_2x_scaler_operand_plus_one = REDUCE_2X_SCALER_NONE;
    }
}

/**
 *
 * @brief Unpacks binary operands to SrcA & SrcB for reduce kernels
 *
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 * @param tile_index_a: The L1 index in the input DFB to read from, tile_index_a -> UNPACKER0 -> SRCA
 * @param tile_index_b: The L1 index in the input DFB to read from, tile_index_b -> UNPACKER1 -> SRCB
 *
 * This function performs unpacking for reduce kernels, the UNPACKER0 unpacks a single tile from the input DFB
 * to srcA and UNPACKER1 unpacks a single face from the input DFB to srcB.
 *
 */
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
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_index_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    WAYPOINT("UABW");
    _llk_unpack_reduce_(l1_tile_index_a, l1_tile_index_b, tensor_shape);
    WAYPOINT("UABD");
}

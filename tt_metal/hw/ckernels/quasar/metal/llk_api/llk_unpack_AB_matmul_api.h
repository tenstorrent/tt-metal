// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_matmul.h"
#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

/**
* @brief Initialize unpacker for matrix multiply

* @tparam TRANSPOSE_EN: Enables transpose of a tile
* @param operandA: The input0 operand circular buffer
* @param operandB: The input1 operand circular buffer
* @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
* @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
* @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
*
* This function initializes the unpacker to unpack operand 0 from the input0 operand circular buffer into SrcB
* and operand 1 from the input1 operand circular buffer into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
*
* Each operand gets a BFD id allocated from the unpack partition and its table entry is programmed here;
* the DFB ids are used only to fetch buffer info, never as BFD ids. Mind the role flip: operandA feeds
* UNPACR1 -> SrcB (Unp1) and operandB feeds UNPACR0 -> SrcA (Unp0).
*/
template <bool TRANSPOSE_EN = false>
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (UNPACR1)
    // In1 -> srcA (UNPACR0)
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // _llk_unpack_matmul_ takes no TensorShape, so it does not scale its L1 tile indices by the face
    // count; Quasar matmul is full-tile only (tt-metal #45208).
    LLK_ASSERT(
        get_operand_tensor_shape(operandA_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "this path indexes L1 in whole tiles, so it supports full 32x32 tiles only");
    LLK_ASSERT(
        get_operand_tensor_shape(operandB_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "this path indexes L1 in whole tiles, so it supports full 32x32 tiles only");

    llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp1>(operandA_id);
    llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operandB_id);

    _llk_unpack_matmul_init_<TRANSPOSE_EN>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
        ct_dim,
        rt_dim,
        kt_dim);
}

/**
 * @brief Runtime-transpose overload, signature-compatible with the Wormhole/Blackhole llk_api.
 *
 * Quasar takes transpose as a template argument, but the shared Compute API passes it as a runtime
 * value. This overload absorbs that difference here rather than forcing an arch branch into the
 * Compute API. Every parameter is required: the template overload above covers the shorter argument lists.
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 * @param transpose: Transpose flag; only 0 is supported on Quasar (transpose of SrcA is not implemented)
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    llk_unpack_AB_matmul_init<false /*TRANSPOSE_EN*/>(operandA, operandB, ct_dim, rt_dim, kt_dim);
}

/**
 *
 * @brief Performs unpack operation for matrix multiply such that:
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 * @param tile_index_a: The index into the input0 CB (UNPACKER1 -> SRCB)
 * @param tile_index_b: The index into the input1 CB (UNPACKER0 -> SRCA)
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 *
 * This function unpacks input0 and input1 operands from the input circular buffers to the src registers such that:
 * Input 0 -> unpack to SrcB
 * Input 1 -> unpack to SrcA
 * The matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 */
inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB
    // In1/InB -> srcA
    LLK_TDMA_GUARD_NOTE_TDMA(operandA);  // TEN-4746: real unpack (UNPACR) disarms these dfbs
    LLK_TDMA_GUARD_NOTE_TDMA(operandB);

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);

    const std::uint32_t l1_tile_idx_0 =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_idx_1 =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    WAYPOINT("UPMW");
    _llk_unpack_matmul_(ct_dim, rt_dim, kt_dim, l1_tile_idx_0, l1_tile_idx_1);
    WAYPOINT("UPMD");
}

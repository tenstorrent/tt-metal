// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 * Derives the tensor shape from operand_A to support non-default tile dimensions.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for SrcB; one of {NONE, ROW, COL, SCALAR}.
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 *     when input is Tf32 format. Only applicable for ELWMUL operations.
 * @tparam binary_reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB
 * @param operand_A: Logical dataflow buffer id for input A, used to derive the tensor / tile shape
 * @param operand_B: Unused.
 * @param acc_to_dest: Flag to control if the result should be accumulated with the current dest.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(
    const std::uint32_t operand_A,
    [[maybe_unused]] const std::uint32_t operand_B,
    const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    constexpr auto effective_math_fidelity = get_effective_math_fidelity<eltwise_binary_type, math_fidelity>();
    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, effective_math_fidelity, binary_reuse_dest>(
        tensor_shape, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = true) {
    // DPRINT("llk_math_eltwise_binary: dst_index = {}, max dest tiles = {}\n",
    //     dst_index,
    //     get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>());

    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "llk_math_eltwise_binary: dst index exceeds available dest register capacity. Uncomment the DPRINT "
        "block above and enable DPRINT support to inspect the dst index and max dest tile values.");

    constexpr auto effective_math_fidelity = get_effective_math_fidelity<eltwise_binary_type, math_fidelity>();
    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        effective_math_fidelity,
        binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE, dst_index, clear_fp32_dst_acc);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A, const std::uint32_t operand_B, uint dst_index, const bool clear_fp32_dst_acc) {
    // DPRINT("llk_math_eltwise_binary: dst_index = {}, max dest tiles = {}\n",
    //     dst_index,
    //     get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>());

    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "llk_math_eltwise_binary: dst index exceeds available dest register capacity. Uncomment the DPRINT "
        "block above and enable DPRINT support to inspect the dst index and max dest tile values.");

    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    constexpr auto effective_math_fidelity = get_effective_math_fidelity<eltwise_binary_type, math_fidelity>();
    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        effective_math_fidelity,
        binary_reuse_dest>(tensor_shape, dst_index, clear_fp32_dst_acc);
}

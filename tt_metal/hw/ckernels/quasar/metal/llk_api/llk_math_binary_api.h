// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 * Assumes default 32x32 tile shape.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for SrcB. Currently only BroadcastType::NONE is supported on Quasar.
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 *     when input is Tf32 format. Only applicable to ELWMUL.
 * @tparam binary_reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB
 * @tparam enable_direct_indexing: When true, uses direct indexing eltwise instructions
 * @param acc_to_dest: Currently unused on Quasar. Will be adding support for accumulation to dest.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool enable_direct_indexing = false>
inline void llk_math_eltwise_binary_init([[maybe_unused]] const std::uint32_t acc_to_dest = 0) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that acc_to_dest is unused
    _llk_math_eltwise_binary_init_<eltwise_binary_type, math_fidelity, enable_direct_indexing, binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 * Derives the tensor shape from operand_A to support non-default tile dimensions.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for SrcB. Currently only BroadcastType::NONE is supported on Quasar.
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 *     when input is Tf32 format. Only applicable to ELWMUL.
 * @tparam binary_reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB
 * @tparam enable_direct_indexing: When true, uses direct indexing eltwise instructions
 * @param operand_A: Logical dataflow buffer id for input A, used to derive the tensor shape
 * @param operand_B: Unused on Quasar. Present for API compatibility.
 * @param acc_to_dest: Currently unused on Quasar. Will be adding support for accumulation to dest.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool enable_direct_indexing = false>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, [[maybe_unused]] const std::uint32_t operand_B, [[maybe_unused]] const std::uint32_t acc_to_dest = 0) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that acc_to_dest is unused
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape_A = get_operand_tensor_shape(operand_id);

    _llk_math_eltwise_binary_init_<eltwise_binary_type, math_fidelity, enable_direct_indexing, binary_reuse_dest>(tensor_shape_A);
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in the destination register.
 * Assumes default tile shape (32x32) or 4 faces.
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>. Unused tparam; only for API compatibiliy.
 * @tparam src_b_bcast_type: Broadcast type for SrcB. Currently only BroadcastType::NONE is supported on Quasar.
 * @tparam is_fp32_dest_acc_en: Unused tparam; only for API compatibiliy.
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 *     when input is Tf32 format. Unused tparam; only for API compatibiliy.
 * @tparam binary_reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB.
 *     The MOVD2A/B instruction copies a face from dest to the source register before each MOP run.
 * @param dst_index: Tile index into the destination register
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param clear_fp32_dst_acc: Unused param; only for API compatibiliy.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(uint dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc = true) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that clear_fp32_dst_acc is unused
    WAYPOINT("MBIW");
    _llk_math_eltwise_binary_<binary_reuse_dest>(dst_index);
    WAYPOINT("MBID");
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in the destination register.
 * Derives num_faces from operand_A to support non-default tile dimensions.
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>. Unused tparam; only for API compatibiliy.
 * @tparam src_b_bcast_type: Broadcast type for SrcB. Currently only BroadcastType::NONE is supported on Quasar.
 * @tparam is_fp32_dest_acc_en: Unused tparam; only for API compatibiliy.
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 *     when input is Tf32 format. Unused tparam; only for API compatibiliy.
 * @tparam binary_reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB.
 *     The MOVD2A/B instruction copies a face from dest to the source register before each MOP run.
 * @param operand_A: Logical dataflow buffer id for input A, used to derive the number of faces
 * @param operand_B: Unused param; only for API compatibiliy.
 * @param dst_index: Tile index into the destination register
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param clear_fp32_dst_acc: Unused param; only for API compatibiliy.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A, [[maybe_unused]] const std::uint32_t operand_B, uint dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that clear_fp32_dst_acc is unused
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    WAYPOINT("MBIW");
    _llk_math_eltwise_binary_<binary_reuse_dest>(dst_index, num_faces);
    WAYPOINT("MBID");
}

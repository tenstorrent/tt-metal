// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

/**
 * @brief Configure the math (FPU/matrix engine) thread for a matmul: programs address mods and the MVMUL MOP.
 *
 * Derives the in0/in1 tile dimensions and the partial-face flag from the operands' circular buffers,
 * then computes D = in0 * in1 (in0 -> SrcB, in1 -> SrcA).
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level; 0 disables throttling, valid throttled range is
 * {1,2,3,4,5}.
 * @param operandA: Circular-buffer index of in0.
 * @param operandB: Circular-buffer index of in1.
 * @param transpose: Non-zero to transpose in1 faces during the multiply.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @note On the unpack thread, pair with @ref llk_unpack_AB_matmul_init which feeds SrcA/SrcB.
 * @ref llk_math_matmul runs the configured matmul with matching template args.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

/**
 * @brief Perform a matmul block, accumulating in0 * in1 into the destination register.
 *
 * Iterates over the output block reusing SrcA or SrcB (whichever dimension is larger) to minimize reloads.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level; must match the value used at init.
 * @tparam num_faces: Number of faces per tile; only 4 is supported.
 * @param dst_index: Base tile index into the destination register for the output block.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @note Call @ref llk_math_matmul_init with matching template args before this function.
 * @note On the unpack thread, @ref llk_unpack_AB_matmul must feed the operand tiles into SrcA/SrcB.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, uint32_t num_faces = 4 /*not used*/>
inline void llk_math_matmul(const uint dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    static_assert(num_faces == 4, "num_faces other than 4 is not supported in llk_math_matmul");
    LLK_ASSERT(
        (ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "llk_math_matmul: computed matmul dest tile range exceeds available dest register "
        "capacity. Uncomment the DPRINT block below and enable DPRINT support to inspect "
        "the calculated and max dest tile values.");

    // DPRINT("llk_math_matmul: calculated dest tiles = {}, max dest tiles = {} (dst_index={}, ct_dim={},
    // rt_dim={})\n",
    //     ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim),
    //     get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>(),
    //     dst_index,
    //     ct_dim,
    //     rt_dim);

    _llk_math_matmul_<math_fidelity, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}

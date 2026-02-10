// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

/**
*
* @brief Initialize matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA

* @tparam NUM_FIDELITY_PHASES: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when
math is in Fp32 format
* @param ct_dim: number of tiles in the column dimension for a matrix multiply
* @param rt_dim: number of tiles in the row dimension for a matrix multiply
*
* This function initializes the matrix multiply operation where:
* Input 0 * Input 1 -> SrcB * SrcA
* Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]
* Output is a matrix block of dimension [rt_dim, ct_dim]
*/
template <std::uint8_t NUM_FIDELITY_PHASES>
inline void llk_math_matmul_init(const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    _llk_math_matmul_init_<static_cast<ckernel::MathFidelity>(NUM_FIDELITY_PHASES)>(ct_dim, rt_dim);
}

/**
 * @brief Performs matrix multiply operation, where Input 0, Input 1 and Output are each 1 tile
 *
 * @param dst_index: Tile index into the destination register
 *
 * This function performs the matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA,
 * Input 0 = 1 tile -> SrcB reg, Input 1 = 1 tile -> SrcA reg,
 * Output = 1 tile -> Dst reg at specified dst_index
 */
inline void llk_math_matmul_tile(const std::uint32_t dst_index) { _llk_math_matmul_tile_(dst_index); }

/**
 *
 * @brief Performs matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA, where
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 *
 * @param ct_dim: number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: number of tiles in the row dimension for a matrix multiply
 *
 * This function performs the matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA,
 * Input 0 dim = [rt_dim, 1] -> SrcB reg, Input 1 dim = [1, ct_dim] -> SrcA reg
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 * This function does not iterate over kt_dim, must iterate over kt_dim externally to this function
 * Dest index is always assumed to start at 0 for this operation
 *
 */
inline void llk_math_matmul_block(const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    _llk_math_matmul_block_(ct_dim, rt_dim);
}

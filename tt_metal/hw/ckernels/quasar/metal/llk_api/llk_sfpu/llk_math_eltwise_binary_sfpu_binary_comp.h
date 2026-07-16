// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_assert.h"
#include "sfpu/ckernel_sfpu_binary_comp.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for elementwise integer greater-than compare
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer compare)
 * @tparam DATA_FORMAT: Data format of the integer operands
 */
template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_gt_int_init() {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU gt_int currently supports Int32 only");
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Performs elementwise integer greater-than: y = (x0 > x1) ? 1 : 0
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer compare)
 * @tparam DATA_FORMAT: Data format of the integer operands
 * @tparam ITERATIONS: Number of iterations for given face
 * @tparam SIGN_MAGNITUDE_FORMAT: Sign-magnitude Int32 encoding for operands and result
 *
 * @param idst0: The index of the tile in DST register buffer to use as first operand
 * @param idst1: The index of the tile in DST register buffer to use as second operand
 * @param odst: The index of the tile in DST register buffer to use as output
 * @param vector_mode: Vector mode (must be VectorMode::RC)
 */
template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_gt_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU gt_int currently supports Int32 only");
    LLK_ASSERT(
        vector_mode == VectorMode::R || vector_mode == VectorMode::C || vector_mode == VectorMode::RC ||
            vector_mode == VectorMode::None,
        "Quasar SFPU gt_int only supports vector modes R, C, RC, None");

    constexpr std::uint32_t tile_stride = NUM_FACES * FACE_R_DIM;
    const std::uint32_t in0_offset = idst0 * tile_stride;
    const std::uint32_t in1_offset = idst1 * tile_stride;
    const std::uint32_t out_offset = odst * tile_stride;

    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::gt, SIGN_MAGNITUDE_FORMAT>,
        in0_offset,
        in1_offset,
        out_offset,
        vector_mode);
}

}  // namespace ckernel

#endif  // ARCH_QUASAR

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_assert.h"
#include "sfpu/ckernel_sfpu_add.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for elementwise integer add
 */
inline void llk_math_eltwise_binary_sfpu_add_int_init() { _llk_math_eltwise_sfpu_init_(); }

/**
 * @brief Performs elementwise integer add: y = add(x0, x1)
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer add)
 * @tparam ITERATIONS: Number of iterations for given face
 * @tparam DATA_FORMAT: Data format of the integer operands
 * @tparam SIGN_MAGNITUDE_FORMAT: Sign-magnitude Int32 encoding for operands and result
 *
 * @param idst0: The index of the tile in DST register buffer to use as first operand
 * @param idst1: The index of the tile in DST register buffer to use as second operand
 * @param odst: The index of the tile in DST register buffer to use as output
 * @param vector_mode: Vector mode (must be VectorMode::RC)
 */
template <bool APPROXIMATE, int ITERATIONS, DataFormat DATA_FORMAT, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_add_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    LLK_ASSERT(
        vector_mode == VectorMode::R || vector_mode == VectorMode::C || vector_mode == VectorMode::RC ||
            vector_mode == VectorMode::None,
        "Quasar SFPU add_int only supports vector modes R, C, RC, None");
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU add_int currently supports Int32 only");
    constexpr std::uint32_t tile_stride = NUM_FACES * FACE_R_DIM;
    const std::uint32_t in0_offset = idst0 * tile_stride;
    const std::uint32_t in1_offset = idst1 * tile_stride;
    const std::uint32_t out_offset = odst * tile_stride;

    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::_add_int_<APPROXIMATE, ITERATIONS, DATA_FORMAT, 0, SIGN_MAGNITUDE_FORMAT>,
        in0_offset,
        in1_offset,
        out_offset,
        vector_mode);
}

}  // namespace ckernel

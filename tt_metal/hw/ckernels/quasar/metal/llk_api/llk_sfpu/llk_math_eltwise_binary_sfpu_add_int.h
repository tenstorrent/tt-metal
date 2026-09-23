// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_add_int.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for elementwise integer add
 */
inline void llk_math_eltwise_binary_sfpu_add_int_init() {
    sfpu::AddInt<false, DataFormat::Int32, DST_SYNC_MODE, DST_ACCUM_MODE>::init();
}

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
 * @param vector_mode: Vector mode, values = <R/C/RC/None>
 */
template <bool APPROXIMATE, int ITERATIONS, DataFormat DATA_FORMAT, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_add_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU add_int currently supports Int32 only");
    sfpu::AddInt<APPROXIMATE, DATA_FORMAT, DST_SYNC_MODE, DST_ACCUM_MODE, SIGN_MAGNITUDE_FORMAT, ITERATIONS>::calculate(
        idst0, idst1, odst, vector_mode);
}

}  // namespace ckernel

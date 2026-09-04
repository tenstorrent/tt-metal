// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_mul_int32.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for elementwise integer multiply
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer mul)
 * @tparam DATA_FORMAT: Data format of the integer operands, values = <Int32>
 * @note Call before @ref llk_math_eltwise_binary_sfpu_mul_int on the same thread.
 */
template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_mul_int_init() {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU mul_int currently supports Int32 only");
    sfpu::MulInt<APPROXIMATE, DATA_FORMAT, DST_SYNC_MODE, DST_ACCUM_MODE>::init();
}

/**
 * @brief Performs elementwise integer multiply: y = mul(x0, x1)
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer mul)
 * @tparam DATA_FORMAT: Data format of the integer operands, values = <Int32>
 * @tparam ITERATIONS: Number of iterations for given face
 * @tparam SIGN_MAGNITUDE_FORMAT: Sign-magnitude Int32 encoding for operands and result
 *
 * @param idst0: The index of the tile in DST register buffer to use as first operand
 * @param idst1: The index of the tile in DST register buffer to use as second operand
 * @param odst: The index of the tile in DST register buffer to use as output
 * @param vector_mode: Vector mode, values = <R/C/RC/None>
 * @note Call @ref llk_math_eltwise_binary_sfpu_mul_int_init with matching template args
 *       before this function.
 */
template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_mul_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU mul_int currently supports Int32 only");
    sfpu::MulInt<APPROXIMATE, DATA_FORMAT, DST_SYNC_MODE, DST_ACCUM_MODE, SIGN_MAGNITUDE_FORMAT, ITERATIONS>::calculate(
        idst0, idst1, odst, vector_mode);
}

}  // namespace ckernel

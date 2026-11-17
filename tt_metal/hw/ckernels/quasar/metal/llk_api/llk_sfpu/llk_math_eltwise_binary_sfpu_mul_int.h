// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_binary_sfpu_macros.h"
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
    _llk_math_eltwise_sfpu_init_();
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
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _mul_int32_,
        (APPROXIMATE, ITERATIONS, SIGN_MAGNITUDE_FORMAT),
        idst0,
        idst1,
        odst,
        vector_mode);
}

/**
 * @brief Initialize SFPU for sign-magnitude fixed-point integer multiply
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer mul)
 * @tparam DATA_FORMAT: Data format of the integer operands, values = <Int32>
 * @tparam SIGN_MAGNITUDE_FORMAT, ENSURE_*: Must match the values passed to the op. They select
 *         whether the SFPLOADMACRO sequence is programmed, so a mismatch leaves the op's fast path
 *         running against an unprogrammed sequence.
 * @note Call before @ref llk_math_eltwise_binary_sfpu_mul_int_sm_shift23 on the same thread.
 */
template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    bool SIGN_MAGNITUDE_FORMAT = true,
    bool ENSURE_POS_A = false,
    bool ENSURE_POS_B = false,
    bool ENSURE_A_23BIT = false,
    bool ENSURE_B_23BIT = false>
inline void llk_math_eltwise_binary_sfpu_mul_int_sm_shift23_init() {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU mul_int sm_shift23 supports Int32 only");
    _llk_math_eltwise_sfpu_init_();
    sfpu::_mul_int32_sm_shift23_init_<
        SIGN_MAGNITUDE_FORMAT,
        ENSURE_POS_A,
        ENSURE_POS_B,
        ENSURE_A_23BIT,
        ENSURE_B_23BIT>();
}

/**
 * @brief Performs elementwise sign-magnitude fixed-point multiply: y = sign * ((|x0| * |x1|) >> 23)
 *
 * x1 is a Q0.23 fixed-point scale, so y is x0 rescaled by x1 / 2^23.
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer mul)
 * @tparam DATA_FORMAT: Data format of the integer operands, values = <Int32>
 * @tparam ITERATIONS: Number of iterations for given face
 * @tparam SIGN_MAGNITUDE_FORMAT: Sign-magnitude Int32 encoding for operands and result
 * @tparam ENSURE_POS_A: Caller guarantees x0 >= 0, dropping its sign handling
 * @tparam ENSURE_POS_B: Caller guarantees x1 >= 0, dropping its sign handling
 * @tparam ENSURE_A_23BIT: Caller guarantees |x0| < 2^23, dropping one cross product
 * @tparam ENSURE_B_23BIT: Caller guarantees |x1| < 2^23, dropping one cross product
 *
 * @param idst0: The index of the tile in DST register buffer to use as first operand
 * @param idst1: The index of the tile in DST register buffer to use as second operand
 * @param odst: The index of the tile in DST register buffer to use as output
 * @param vector_mode: Vector mode, values = <R/C/RC/None>
 * @note Call @ref llk_math_eltwise_binary_sfpu_mul_int_sm_shift23_init with matching template
 *       args before this function. See _mul_int32_sm_shift23_ for the exactness envelope.
 */
template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = true,
    bool ENSURE_POS_A = false,
    bool ENSURE_POS_B = false,
    bool ENSURE_A_23BIT = false,
    bool ENSURE_B_23BIT = false>
inline void llk_math_eltwise_binary_sfpu_mul_int_sm_shift23(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU mul_int sm_shift23 supports Int32 only");
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _mul_int32_sm_shift23_,
        // SFPU_BINARY_CALL forwards the tuple verbatim, so TILE_SHAPE is named here to reach the
        // ensure_* flags behind it. The Compute API is 32x32-only, as mul_int_tile is.
        (APPROXIMATE,
         ITERATIONS,
         SIGN_MAGNITUDE_FORMAT,
         trisc::DstTileShape::Tile32x32,
         ENSURE_POS_A,
         ENSURE_POS_B,
         ENSURE_A_23BIT,
         ENSURE_B_23BIT),
        idst0,
        idst1,
        odst,
        vector_mode);
}

}  // namespace ckernel

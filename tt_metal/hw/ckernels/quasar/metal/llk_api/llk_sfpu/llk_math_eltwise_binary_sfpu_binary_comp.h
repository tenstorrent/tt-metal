// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#ifdef ARCH_QUASAR

#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_binary_comp.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for an elementwise integer compare (lt/gt/le/ge/eq/ne)
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer compare)
 * @tparam DATA_FORMAT: Data format of the integer operands
 */
template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_comp_int_init() {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU integer compares currently support Int32 only");
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Performs an elementwise integer compare: y = (x0 OP x1) ? 1 : 0
 *
 * @tparam APPROXIMATE: Approximation mode (unused for integer compare)
 * @tparam DATA_FORMAT: Data format of the integer operands
 * @tparam OP: Compare, one of SfpuType::lt/gt/le/ge/eq/ne
 * @tparam ITERATIONS: Number of iterations for given face
 * @tparam SIGN_MAGNITUDE_FORMAT: Operands and result are sign-magnitude Int32 (FPU-promoted Int8); native
 *         Int32 is two's complement and keeps the default
 *
 * @param idst0: The index of the tile in DST register buffer to use as first operand
 * @param idst1: The index of the tile in DST register buffer to use as second operand
 * @param odst: The index of the tile in DST register buffer to use as output
 * @param vector_mode: Vector mode, values = <R/C/RC/None>
 */
template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    SfpuType OP,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_comp_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU integer compares currently support Int32 only");
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_int32,
        (APPROXIMATE, ITERATIONS, OP, SIGN_MAGNITUDE_FORMAT),
        idst0,
        idst1,
        odst,
        vector_mode);
}

// Per-compare entry points over llk_math_eltwise_binary_sfpu_comp_int / _comp_int_init.
template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_lt_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_gt_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_le_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_ge_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_eq_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_ne_int_init() {
    llk_math_eltwise_binary_sfpu_comp_int_init<APPROXIMATE, DATA_FORMAT>();
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_lt_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::lt, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_gt_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::gt, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_le_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::le, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_ge_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::ge, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_eq_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::eq, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

template <
    bool APPROXIMATE,
    DataFormat DATA_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_ne_int(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_comp_int<APPROXIMATE, DATA_FORMAT, SfpuType::ne, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(
        idst0, idst1, odst, vector_mode);
}

}  // namespace ckernel

#endif  // ARCH_QUASAR

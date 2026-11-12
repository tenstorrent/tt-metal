// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sin_tile_init() { MATH(SFPU_UNARY_INIT_FN(sine, ckernel::sfpu::sine_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric sine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sin_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sine,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cos_tile_init() { MATH(SFPU_UNARY_INIT_FN(cosine, ckernel::sfpu::cosine_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric cosine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cos_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_cosine,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(acosh, ckernel::sfpu::init_inverse_hyperbolic, (APPROX, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic cosine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_acosh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// Quasar's SfpuType (tt_llk_quasar/llk_lib/llk_defs.h:75) has no `tan`, `asin`, `acos`,
// `atan`, `cosh` or `sinh` enumerator. SFPU_UNARY_INIT_FN names SfpuType::<OP>
// non-dependently, so each of the six *_tile_init bodies below is a hard error on
// Quasar whether or not it is ever instantiated -- which made the whole of this header
// uncompilable on Quasar, and took sin_tile/cos_tile down with it even though
// SfpuType::sine and SfpuType::cosine exist and have working Quasar LLKs. Guarding the
// six broken inits is strictly additive on Quasar (nothing could ever have compiled
// against them) and a no-op on every other architecture.
ALWI void tan_tile_init() { MATH(SFPU_UNARY_INIT_FN(tan, ckernel::sfpu::tangent_init, (APPROX))); }
#endif  // ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of the trigonometric tan operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tan_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_tangent,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(asinh, ckernel::sfpu::init_inverse_hyperbolic, (APPROX, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic sine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_asinh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile_init() { MATH(SFPU_UNARY_INIT_FN(atanh, ckernel::sfpu::init_atanh, (APPROX, is_fp32_dest_acc_en))); }

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic tangent operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_atanh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

// clang-format off
/**
 * Performs element-wise computation of arcsine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_asin,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// ABSENT ON QUASAR -- see the note on tan_tile_init above.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile_init() { MATH(SFPU_UNARY_INIT_FN(asin, sfpu::asin_acos_init, (is_fp32_dest_acc_en))); }
#endif  // ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of arctan on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_atan,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// ABSENT ON QUASAR -- see the note on tan_tile_init above.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(atan, sfpu::atan_init, (true /*APPROXIMATION_MODE*/, is_fp32_dest_acc_en)));
}
#endif  // ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of arcossine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_acos,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// ABSENT ON QUASAR -- see the note on tan_tile_init above.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile_init() { MATH(SFPU_UNARY_INIT_FN(acos, sfpu::asin_acos_init, (is_fp32_dest_acc_en))); }
#endif  // ARCH_QUASAR

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// ABSENT ON QUASAR -- see the note on tan_tile_init above.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile_init() { MATH(SFPU_UNARY_INIT_FN(cosh, ckernel::sfpu::cosh_init, (APPROX, is_fp32_dest_acc_en))); }
#endif  // ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic cosine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_cosh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
#ifndef ARCH_QUASAR
// ABSENT ON QUASAR -- see the note on tan_tile_init above.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile_init() { MATH(SFPU_UNARY_INIT_FN(sinh, ckernel::sfpu::sinh_init, (APPROX, is_fp32_dest_acc_en))); }
#endif  // ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic sine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sinh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

}  // namespace ckernel

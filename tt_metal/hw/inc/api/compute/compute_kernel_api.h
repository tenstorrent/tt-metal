// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#ifndef ARCH_QUASAR
#include "ckernel_globals.h"
#include "ckernel_debug.h"
#endif
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"
#include "internal/risc_attribs.h"

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "llk_math_debug.h"
#endif
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_add_top_row.h"
#include "ckernel_sfpu_max_pool_indices.h"
#include "llk_math_binary_api.h"
#include "llk_math_reduce_api.h"
// SFPU op kernels invoked directly via the unary macros below. The macros
// themselves come from llk_math_eltwise_unary_sfpu_macros.h. These BH/WH-only
// kernels (log, abs, ...) have no Quasar implementation; sigmoid/silu are
// shared and included for Quasar in the #else branch below.
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_tanh.h"
#include "ckernel_sfpu_signbit.h"
#include "ckernel_sfpu_abs.h"
#include "ckernel_sfpu_sign.h"
#include "ckernel_sfpu_square.h"
#include "ckernel_sfpu_tiled_prod.h"
#include "ckernel_sfpu_unary_power.h"
#include "ckernel_sfpu_exp2.h"
#include "ckernel_sfpu_heaviside.h"
#include "ckernel_sfpu_expm1.h"
#include "ckernel_sfpu_topk.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "ckernel_sfpu_reduce.h"
#include "ckernel_sfpu_alt_complex_rotate90.h"
#else
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_tanh.h"
#include "ckernel_sfpu_square.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_add_int.h"
#include "llk_math_eltwise_binary_sfpu_mul_int.h"
#include "llk_math_eltwise_binary_sfpu_binary_comp.h"
#include "ckernel_sfpu_topk.h"
#endif
#define MATH(...) __VA_ARGS__
#else
#define MATH(...)
#endif

#ifdef TRISC_PACK
#include "llk_io_pack.h"
#ifndef ARCH_QUASAR
// Pack-thread SFPU op kernels invoked via the unary macros (silu/tanh/sigmoid
// *_tile_pack helpers below). Quasar is out of scope for the unary macro
// refactor, so these are BH/WH only.
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_tanh.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#define PACK(...) __VA_ARGS__
#else
#define PACK(...)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#ifndef ARCH_QUASAR
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#endif
#include "llk_io_unpack.h"
#define UNPACK(...) __VA_ARGS__
#else
#define UNPACK(...)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void sigmoid_tile_init() {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(sigmoid));
#else
    MATH(SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (fast_and_approx)));
#endif
}

// clang-format off
/**
 * Performs element-wise computation of sigmoid on each element of a tile
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
template <VectorMode vec_mode = VectorMode::RC, bool fast_and_approx = false>
ALWI void sigmoid_tile(uint32_t idst) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid, (8 /*ITERATIONS*/), idst, vec_mode));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sigmoid,
        (fast_and_approx, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        vec_mode));
#endif
}

// clang-format off
/**
 * Performs SILU (same as Swish) operation on each element of a tile
 * in DST register at index tile_index. Uses the following implementation:
 * Silu[x] = x*Sigmoid[x]
 *
 * Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void silu_tile(uint32_t idst) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (8 /*ITERATIONS*/), idst, ::ckernel::VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, 8 /* ITERATIONS */), idst, VectorMode::RC));
#endif
}

ALWI void silu_tile_init() {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(silu));
#else
    MATH(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (APPROX)));
#endif
}

// TODO: Move to trigonometry.h (https://github.com/tenstorrent/tt-metal/issues/47942)
/**
 * Please refer to documentation for any_init.
 *
 * If using fast and approximate implementation of tanh_tile(), then tanh_tile_init() should be also be called with
 * fast_and_approx = true.
 */
template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT_FN(tanh, sfpu::tanh_init, (fast_and_approx, DST_ACCUM_MODE)));
#else
    MATH(SFPU_UNARY_INIT(tanh));
#endif
}

// TODO: Move to trigonometry.h (https://github.com/tenstorrent/tt-metal/issues/47942)
// clang-format off
/**
 * Performs element-wise computation of tanh on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * If using fast and approximate mode, then tanh_tile_init() should be also be called with fast_and_approx = true beforehand.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx | Whether to use fast and approximate mode                                   | bool     | True or False                                         | False    |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_tanh,
        (fast_and_approx, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (8 /* ITERATIONS */), idst, ::ckernel::VectorMode::RC));
#endif
}

// clang-format off
/**
 * Performs element-wise computation of square value on each element of a tile
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
ALWI void square_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (APPROX), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (SFPU_ITERATIONS), idst, VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void square_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(square));
#else
    MATH(SFPU_UNARY_INIT(square, sfpu::init_square));
#endif
}

#ifndef ARCH_QUASAR

template <bool fast_and_approx = false>
ALWI void sigmoid_tile_init_pack() {
    PACK(SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (fast_and_approx)));
}

template <VectorMode vec_mode = VectorMode::RC, bool fast_and_approx = false>
ALWI void sigmoid_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sigmoid,
        (fast_and_approx, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        vec_mode));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void log_tile_init() {
    // TODO(AP): move out init
    MATH(SFPU_UNARY_INIT_FN(log, sfpu::log_init, (APPROX, fast_and_approx, DST_ACCUM_MODE)));
}

// clang-format off
/**
 * Performs element-wise computation of logarithm on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Note: The base scale is the bit representation of the inverse of the log base.
 * e.g. 1/ln(2) for log2(x) is 0x3fb8aa3b.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void log_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_log,
        (APPROX, fast_and_approx, false /* HAS_BASE_SCALING */, DST_ACCUM_MODE),
        idst,
        VectorMode::RC,
        0));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void log_with_base_tile_init() {
    // TODO(AP): move out init
    MATH(SFPU_UNARY_INIT_FN(log_with_base, sfpu::log_init, (APPROX, fast_and_approx, DST_ACCUM_MODE)));
}

// clang-format off
/**
 * Performs element-wise computation of logarithm with a specified base on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | base_scale      | Bit representation of Inverse of log base e.g. 1/ln(2) to compute log2(x)  | uint32_t | Positive integers                                     | True     |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_log,
        (APPROX, fast_and_approx, true /* HAS_BASE_SCALING */, DST_ACCUM_MODE),
        idst,
        VectorMode::RC,
        base_scale));
}

template <bool fast_and_approx = false>
ALWI void tanh_tile_init_pack() {
    PACK(SFPU_UNARY_INIT_FN(tanh, sfpu::tanh_init, (fast_and_approx, DST_ACCUM_MODE)));
}

template <bool fast_and_approx = false>
ALWI void tanh_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_tanh,
        (fast_and_approx, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void signbit_tile_init() {
    MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_init));
}

// clang-format off
/**
 * Sets the sign bit of each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void signbit_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit, (APPROX, 8 /* ITERATIONS */), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void signbit_tile_int32_init() {
    MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_int32_init));
}

// clang-format off
/**
 * Sets the sign bit of each element of a tile (int32 datatype)
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void signbit_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit_int32, (APPROX, 8 /* ITERATIONS */), idst, VectorMode::RC));
}

// clang-format off
/**
 * Performs element-wise computation of absolute value on each element of a tile
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
ALWI void abs_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void abs_tile_init() { MATH(SFPU_UNARY_INIT(abs)); }

// clang-format off
/**
 * Performs element-wise computation of absolute value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Note: This version of the function is for int32 datatype
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void abs_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs_int32, (APPROX), idst, VectorMode::RC));
}

// clang-format off
/**
 * Will store in the output of the compute core the signum of the tile.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sign_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sign, (APPROX), idst, VectorMode::RC, 1 /* exponent_size_8 */));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sign_tile_init() { MATH(SFPU_UNARY_INIT(sign)); }

// clang-format off
/**
 * Performs element-wise multiplication on each row of a tile.
 * The DST register buffer must be in
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
ALWI void tiled_prod_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tiled_prod, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void tiled_prod_tile_init() { MATH(SFPU_UNARY_INIT(tiled_prod)); }

// POWER : y = x^(const param0)
// clang-format off
/**
 * Performs element-wise computation of power operation (x ^(const param0)) value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The exponent as IEEE 754 float bits                                        | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void power_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_power,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void power_tile_init() {
    MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::power>(ckernel::sfpu::sfpu_unary_pow_init));
}

// POWER_ITERATIVE : y = x^(const param0)
// clang-format off
/**
 * Performs element-wise computation of power operation (x ^(const param0)) value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Note: Unlike power_tile, power_iterative_tile() only supports positive integer scalars. It uses an iterative multiplication loop to compute values, and is faster than power_tile for small exponents (e.g. 1, 2, 3)
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The integer exponent value                                                 | uint32_t | Must be a non-negative integer exponent               | True     |
 */
// clang-format on
ALWI void power_iterative_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_power_iterative,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void power_iterative_tile_init() { MATH(SFPU_UNARY_INIT(power)); }

// clang-format off
// exp2 : y = 2 ^ x  ==> [y = exp(x * log(2))]
/**
 * Performs element-wise computation of 2^x value where x is each element of a tile
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
ALWI void exp2_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exp2, (true /* APPROXIMATE */, DST_ACCUM_MODE), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void exp2_tile_init() { MATH(SFPU_UNARY_INIT_FN(exp2, sfpu::exp2_init, (true /*APPROXIMATE*/, DST_ACCUM_MODE))); }

// heaviside : y = 0 if x < 0 , 1 if x > 0 , else value
// clang-format off
/**
 * Performs element-wise computation of:  y = 0 if x < 0 , 1 if x > 0 , y= value  where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value the output is if the input is greater than 0                     | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void heaviside_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_heaviside, (APPROX), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void heaviside_tile_init() { MATH(SFPU_UNARY_INIT(heaviside)); }

// expm1 : (exp(x) - 1)
// clang-format off
/**
 * Performs element-wise computation of exp(x) - 1, v where x is each element of a tile
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
template <bool approx = false>
ALWI void expm1_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_expm1,
        (approx, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool approx = false>
ALWI void expm1_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(expm1, sfpu::expm1_init, (approx, DST_ACCUM_MODE)));
}

ALWI void silu_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, 8 /* ITERATIONS */), idst, VectorMode::RC));
}
ALWI void silu_tile_init_pack() { PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (APPROX))); }

#endif  // !ARCH_QUASAR — TopK below is all-arch

// topK local sort
// clang-format off
/**
 * Performs local sort stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The algorithm used to implement TopK is found here:
 * https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
 *
 * The local sort stage sorts the data into length-K subsequences of
 * alternating directions, in place. If i_start_phase != i_end_phase, all
 * phases in the range i_start_phase to i_end_phase (inclusive) are computed.
 * If i_start_phase == i_end_phase, only that phase is computed, with
 * i_start_step and i_end_step defining how many steps are computed. This can
 * be used to extend the OP support for cases when K > 64.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that local sort is done across columns on 64 values spanning across
 * 2 tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idir            | The sorting direction of the local sort (0 == decreasing, 1 == increasing) | int32    | 0 to 1                                                | True     |
 * | i_end_phase     | The end phase of the local sort (should be set to log(K)-1)                | int32    | 1 to 5                                                | True     |
 * | i_start_phase   | The start phase of the local sort (should be set to 0)                     | int32    | 0 to 5                                                | False    |
 * | i_end_step      | The end step to perform if i_start_phase == i_end_phase                    | int32    | 4 to 6                                                | False    |
 * | i_start_step    | The start step to perform if i_start_phase == i_end_phase                  | int32    | 4 to 6                                                | False    |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 */
// clang-format on
template <bool stable_sort = false>
ALWI void topk_local_sort(
    uint32_t idst, int idir, int i_end_phase, int i_start_phase = 0, int i_end_step = 0, int i_start_step = 0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_phases_steps,
        (true /* APPROXIMATE */, DST_ACCUM_MODE, stable_sort),
        idst,
        VectorMode::RC_custom,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step));
}

// topK merge
// clang-format off
/**
 * Performs merge stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The merge stage combines length-K subsequences that are 2^m_iter apart,
 * such that the first subsequence receives the top K values, and the
 * second subsequence receives the bottom K values.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that merge is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | m_iter          | The index of the merge & rebuild iteration of the algorithm                | int32    | 0 to 9                                                | True     |
 * | k               | The number of sorted values to return                                      | int32    | {4, 8, 16, 32, 64}                                    | True     |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 */
// clang-format on
template <bool idir = false, bool stable_sort = false>
ALWI void topk_merge(uint32_t idst, int m_iter, int k) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_merge,
        (true /* APPROXIMATE */, DST_ACCUM_MODE, idir, stable_sort),
        idst,
        VectorMode::RC_custom,
        m_iter,
        k));
}

// topK rebuild
// clang-format off
/**
 * Performs rebuild stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The rebuild stage sorts the length-K subsequences that are 2^(m_iter+1)
 * apart, such that they alternate directions.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that rebuild is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idir            | The sorting direction of the local sort (0 == decreasing, 1 == increasing) | bool     | 0 to 1                                                | True     |
 * | m_iter          | The index of the merge & rebuild iteration of the algorithm                | int32    | 0 to 9                                                | True     |
 * | k               | The number of sorted values to return                                      | int32    | {4, 8, 16, 32, 64}                                    | True     |
 * | logk            | The log of K                                                               | int32    | 2 to 6                                                | True     |
 * | skip_second     | Whether or not to skip second tile                                         | int32    | 0 to 1                                                | True     |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 */
// clang-format on
template <bool stable_sort = false>
ALWI void topk_rebuild(uint32_t idst, bool idir, int m_iter, int k, int logk, int skip_second) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_rebuild,
        (true /* APPROXIMATE */, DST_ACCUM_MODE, stable_sort),
        idst,
        VectorMode::RC_custom,
        idir,
        m_iter,
        k,
        logk,
        skip_second));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void topk_tile_init() { MATH(SFPU_UNARY_INIT_FN(topk_local_sort, sfpu::topk_init, (true /* APPROXIMATE */))); }

#ifndef ARCH_QUASAR  // BH/WH-only ops below

// clang-format off
/**
 * Performs MaxPool with indices algorithm on the data tile and index tile
 * that are pre-loaded in DST register. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * | Argument        | Description                                                                 | Type       | Valid Range                                           | Required |
 * |-----------------|-----------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register containing the data to be reduced     | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | idst_idx        | The index of the tile in DST register containing the indices of the data    | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | chunk           | The index of the intra-kernel "chunk" of data for large kernel accumulation | uint32_t   | 0 to UINT_MAX                                         | False    |
 * | num_rows        | The number of rows to use for the MaxPool operation                         | uint32_t   | <= 32, but note either 9 or 32 rows will be reduced   | False    |
 * | layout          | The data layout of the data in DST                                          | DataLayout | TILE or ROW_MAJOR                                     | False    |
 * | accumulate      | Whether to accumulate results for large kernels                             | bool       | true, false                                           | False    |
 * | ITERATIONS      | The number of iterations to perform (unused)                                | int        | 1 to 8                                                | False    |
 */
// clang-format on
template <
    int num_rows = 9,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE,
    bool accumulate = false,
    int ITERATIONS = 8>
ALWI void max_reduce_with_indices(uint32_t idst, uint32_t idst_idx, uint32_t chunk = 0) {
    static_assert(num_rows <= 32, "num_rows must be <= 32");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_max_pool_with_indices,
        (true /* APPROXIMATE */, DST_ACCUM_MODE, num_rows, ITERATIONS, layout, accumulate),
        idst,
        idst_idx,
        0 /* DST out unused, but required for _llk_math_eltwise_binary_sfpu_params_ */,
        VectorMode::RC,
        chunk)));
}

/**
 * Please refer to documentation for any_init.
 */
template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
ALWI void max_reduce_with_indices_init() {
    MATH((SFPU_BINARY_INIT_FN(
        max_pool_with_indices, sfpu::init_max_pool_with_indices, (true /* APPROXIMATE */, layout))));
}

// clang-format off
/**
 * Performs reduce operation (sum, average, max, min) on a 32x32 tile for column reduction and multiple tiles for row reduction, placing output values into the first row.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only 32x32 tile dimensions are supported
 *  - This kernel is optimized for 32x32 tile dimensions and uses VectorMode::RC_custom for customized reduction
 *  - Column reduction (REDUCE_COL) is supported for all pool types; row reduction (REDUCE_ROW) is supported for SUM, MAX and MIN only.
 *  - REDUCE_COL operates on a single tile only (ct_dim = 1, rt_dim = 1).
 *  - REDUCE_ROW supports multiple tiles: ct_dim and rt_dim specify the tile block dimensions to reduce over.
 *
 * | Argument        | Description                                                                     | Type      | Valid Range
 * |-----------------|---------------------------------------------------------------------------------|-----------|-------------------------------------------------------
 * | pool_type       | The type of reduction operation, SUM or AVG (MAX/MIN for REDUCE_COL only)       | PoolType  | SUM, AVG, MAX, MIN
 * | format          | The data format for the reduction operation                                     | DataFormat| Float32, Int32, UInt32, UInt16, Float16_b
 * | reduce_dim      | The reduction dimension                                                         | ReduceDim | REDUCE_COL or REDUCE_ROW (REDUCE_ROW only for SUM and MAX)
 * | idst            | The index of the tile in DST register containing the data to be reduced         | uint32_t  | Must be less than the size of the DST register buffer
 * | ct_dim          | Tile dimension along columns (runtime); must be 1 when reduce_dim is REDUCE_COL | uint32_t  | >= 1; default 1
 * | rt_dim          | Tile dimension along rows (runtime); must be 1 when reduce_dim is REDUCE_COL    | uint32_t  | >= 1; default 1
 */
// clang-format on
template <PoolType pool_type, DataFormat format, ReduceDim reduce_dim = ReduceDim::REDUCE_COL>
ALWI void sfpu_reduce(uint32_t idst, uint32_t ct_dim = 1, uint32_t rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || pool_type == PoolType::MAX || pool_type == PoolType::MIN)),
        "Only column reduction (REDUCE_COL) is supported for all pool types; row reduction (REDUCE_ROW) is only "
        "supported for SUM, MAX and MIN");
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32 ||
            format == DataFormat::UInt16 || format == DataFormat::Float16_b,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32, UInt16, Float16_b");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool type. Supported pool types: SUM, AVG, MAX, MIN");

    // This kernel is optimized for 32x32 tiles and uses RC_custom vector mode for custom reduction
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_reduce,
        (pool_type, reduce_dim, format, DST_ACCUM_MODE),
        idst,
        VectorMode::RC_custom,
        ct_dim,
        rt_dim));
}

/**
 * @brief Initialization for SFPU reduce kernel.
 *        Must be called before sfpu_reduce() to set up the necessary configurations for reduction operations.
 *        The same init is used for both REDUCE_COL and REDUCE_ROW; it does not take tile dimensions.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam format The data format, currently supported: (Float32, Int32, UInt32, UInt16, Float16_b)
 */
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_init() {
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool type. Supported pool types: SUM, AVG, MAX, MIN");
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32 ||
            format == DataFormat::UInt16 || format == DataFormat::Float16_b,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32, UInt16, Float16_b");

    MATH(SFPU_UNARY_INIT_FN_ARGS(reduce, sfpu::init_reduce, (pool_type, format, DST_ACCUM_MODE), 1 /* block_ct_dim */));
}

// clang-format off
/**
 * Performs element-wise add_top_row operation between the top rows of two tiles in DST register.
 * Takes the top row of tile at dst_tile_0 and adds it with the top row of tile at dst_tile_1,
 * storing the result in the top row of tile at dst_tile_out.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only 32x32 tile dimensions are supported.
 * All tile indices must reference valid tiles within the DST register.
 *
 * | Argument        | Description                                                              | Type      | Valid Range                                           | Required |
 * |-----------------|--------------------------------------------------------------------------|-----------|-------------------------------------------------------|----------|
 * | dst_tile_0      | The index of the first tile in DST register                              | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | dst_tile_1      | The index of the second tile in DST register                             | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | dst_tile_out    | The index of the output tile in DST register                             | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | format          | The data format for the add_top_row operation                            | DataFormat| Float32, Int32, UInt32                                | True     |
 */
// clang-format on
template <DataFormat format>
ALWI void sfpu_add_top_row(uint32_t dst_tile_0, uint32_t dst_tile_1, uint32_t dst_tile_out) {
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32");

    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_top_row,
        (format),
        dst_tile_0,
        dst_tile_1,
        dst_tile_out,
        VectorMode::RC_custom)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sfpu_add_top_row_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(add_top_row, sfpu::init_add_top_row))); }

/**
 * Pauses the cores so that the debug interface can be used to inspect the value of the registers.
 *
 * Return value: None
 */
ALWI void dbg_halt() {
    PACK(dbg_thread_halt<PackThreadId>());
    UNPACK(dbg_thread_halt<UnpackThreadId>());
    MATH(dbg_thread_halt<MathThreadId>());
}

/**
 * Resumes the execution of the cores after a debug halt.
 *
 * Return value: None
 */
ALWI void dbg_unhalt() {
    PACK(dbg_thread_unhalt<PackThreadId>());
    UNPACK(dbg_thread_unhalt<UnpackThreadId>());
    MATH(dbg_thread_unhalt<MathThreadId>());
}

// clang-format off
/**
 * Reads the contents of the specified row of the destination register. It reads 8 dwords at a time.
 *
 * | Argument        | Description                                             | Type      | Valid Range  | Required |
 * |-----------------|---------------------------------------------------------|-----------|--------------|----------|
 * | row_addr        | The row address in the destination register to read     | int       |              | True     |
 * | rd_data         | The array of 8 dwords to store the data                 | uint32_t* |              | True     |
 *
 * Return value: None
 */
// clang-format on
ALWI void dbg_read_dest_acc_row(int row_addr, uint32_t* rd_data) {
    MATH((dbg_get_array_row(dbg_array_id::DEST, row_addr, rd_data)));
}

// unary_max : if x > value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x > value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_max_int32_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (true /* IS_MAX */, false /* IS_UINT */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_max_int32_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(unary_max_int32, sfpu::unary_max_min_int32_init, (true /* IS_MAX */, false /* IS_UINT */)));
}

// unary_max : if x > value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x > value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_max_uint32_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (true /* IS_MAX */, true /* IS_UINT */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_max_uint32_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(unary_max_uint32, sfpu::unary_max_min_int32_init, (true /* IS_MAX */, true /* IS_UINT */)));
}

// unary_max : if x > value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x > value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_max_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min,
        (true /* IS_MAX */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_max_tile_init() { MATH(SFPU_UNARY_INIT_FN(unary_max, sfpu::unary_max_min_init, (true /* IS_MAX */))); }

// clang-format off
/**
 * Treats pairs of numbers as complex numbers and rotates them 90 degrees
 * in the complex plane. The operation is performed on a tile
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
ALWI void alt_complex_rotate90_tile(uint32_t idst) {
    MATH(
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_alt_complex_rotate90, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void alt_complex_rotate90_tile_init() { MATH(SFPU_UNARY_INIT(alt_complex_rotate90)); }

// unary_min : if x < value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x < value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_min_int32_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (false /* IS_MAX */, false /* IS_UINT */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_min_int32_tile_init() {
    MATH(
        SFPU_UNARY_INIT_FN(unary_min_int32, sfpu::unary_max_min_int32_init, (false /* IS_MAX */, false /* IS_UINT */)));
}

// unary_min : if x < value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x < value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_min_uint32_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (false /* IS_MAX */, true /* IS_UINT */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_min_uint32_tile_init() {
    MATH(
        SFPU_UNARY_INIT_FN(unary_min_uint32, sfpu::unary_max_min_int32_init, (false /* IS_MAX */, true /* IS_UINT */)));
}

// unary_min : if x < value --> x, else value
// clang-format off
/**
 * Performs element-wise computation of:  result = x if x < value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_min_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min,
        (false /* IS_MAX */, APPROX),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_min_tile_init() { MATH(SFPU_UNARY_INIT_FN(unary_min, sfpu::unary_max_min_init, (false /* IS_MAX */))); }

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
ALWI uint32_t get_compute_special_value_flags() {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_get_compute_special_value_flags()));
    return ret_val;
}

ALWI uint32_t get_compute_special_value_flags_fpu(uint32_t special_value_flags_reg) {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_extract_compute_special_value_flags<true /* isFpu */>(special_value_flags_reg)));
    return ret_val;
}

ALWI uint32_t get_compute_special_value_flags_sfpu(uint32_t special_value_flags_reg) {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_extract_compute_special_value_flags<false /* isFpu */>(special_value_flags_reg)));
    return ret_val;
}

ALWI void clear_compute_special_value_flags() { MATH((llk_math_clear_compute_special_value_flags())); }
#endif

#endif

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_debug.h"
#include "hostdevcommon/kernel_structs.h"
#include "internal/risc_attribs.h"

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_sfpu_api.h"
#include "llk_math_binary_sfpu_api.h"
#include "llk_math_reduce_api.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#include "llk_io_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#include "llk_io_unpack.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void sigmoid_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid_init<fast_and_approx>()));
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
template <int vec_mode = VectorMode::RC, bool fast_and_approx = false>
ALWI void sigmoid_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<fast_and_approx, DST_ACCUM_MODE>(idst, vec_mode)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void log_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_init<APPROX, fast_and_approx, DST_ACCUM_MODE>()));  // TODO(AP): move out init
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
    MATH((llk_math_eltwise_unary_sfpu_log<APPROX, fast_and_approx, DST_ACCUM_MODE>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX, fast_and_approx, DST_ACCUM_MODE>()));  // TODO(AP):
                                                                                                        // move out init
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
 * | base_scale      | Bit representation of Inverse of log base e.g. 1/ln(2) to compute log2(x)  | uint32_t | Postive integers                                      | True     |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX, fast_and_approx, DST_ACCUM_MODE>(idst, base_scale)));
}

// TODO: Move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 *
 * If using fast and approximate implementation of tanh_tile(), then tanh_tile_init() should be also be called with
 * fast_and_approx = true.
 */
template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()));  // TODO(AP): move out init
}

// TODO: Move to trigonometry.h
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
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void signbit_tile_init() { MATH((llk_math_eltwise_unary_sfpu_signbit_init<APPROX>())); }

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
ALWI void signbit_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_signbit<APPROX>(idst))); }

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
ALWI void signbit_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_signbit_int32<APPROX>(idst))); }

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
ALWI void abs_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_abs<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void abs_tile_init() { MATH((llk_math_eltwise_unary_sfpu_abs_init<APPROX>())); }

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
ALWI void abs_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_abs_int32<APPROX>(idst))); }

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
ALWI void sign_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_sign<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void sign_tile_init() { MATH((llk_math_eltwise_unary_sfpu_sign_init<APPROX>())); }

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
ALWI void square_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_square<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void square_tile_init() { MATH((llk_math_eltwise_unary_sfpu_square_init<APPROX>())); }

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
ALWI void tiled_prod_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_tiled_prod<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void tiled_prod_tile_init() { MATH((llk_math_eltwise_unary_sfpu_tiled_prod_init<APPROX>())); }

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
 * | param0          | The value of the exponent in the power operation                           | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void power_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_power<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void power_tile_init() { MATH((llk_math_eltwise_unary_sfpu_power_init<APPROX>())); }

// clang-format off
// MAX : y = max(idst0, idst1)
/**
 * Performs element-wise computation of max value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * TODO: fix idst1.
 * currently idst1 is not used and (idst0 + 1) is used for max.
 * because don't know how to use 2 dst register with sfpu.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode     | Controls face tiles which are operated on. Default RC - all face tiles.    | int      | Valid members of VectorMode enum                      | False    |
 */
// clang-format on
ALWI void max_tile(uint32_t idst0, uint32_t idst1, int vector_mode = (int)VectorMode::RC) {
    MATH((llk_math_eltwise_unary_sfpu_max<APPROX>(idst0, vector_mode)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void max_tile_init() { MATH((llk_math_eltwise_unary_sfpu_max_init<APPROX>())); }

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
ALWI void exp2_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_exp2<true, DST_ACCUM_MODE>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void exp2_tile_init() { MATH((llk_math_eltwise_unary_sfpu_exp2_init<true>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_heaviside<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void heaviside_tile_init() { MATH((llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_expm1<approx, DST_ACCUM_MODE>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool approx = false>
ALWI void expm1_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_expm1_init<approx, DST_ACCUM_MODE>()));
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
ALWI void silu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst))); }

ALWI void silu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_silu_init<APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_topk_local_sort<true, DST_ACCUM_MODE, stable_sort>(
        idst, idir, i_end_phase, i_start_phase, i_end_step, i_start_step)));
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
    MATH((llk_math_eltwise_unary_sfpu_topk_merge<true, DST_ACCUM_MODE, idir, stable_sort>(idst, m_iter, k)));
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
    MATH((llk_math_eltwise_unary_sfpu_topk_rebuild<true, DST_ACCUM_MODE, stable_sort>(
        idst, idir, m_iter, k, logk, skip_second)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void topk_tile_init() { MATH((llk_math_eltwise_unary_sfpu_topk_init<true>())); }

// clang-format off
/**
 * Performs MaxPool with indices algorithm on the data tile and index tile
 * that are pre-loaded in DST register. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only a reduction of 9 rows is supported at this time.
 *
 * | Argument        | Description                                                              | Type       | Valid Range                                           | Required |
 * |-----------------|--------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register containing the data to be reduced  | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | idst_idx        | The index of the tile in DST register containing the indices of the data | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | num_rows        | The number of rows to use for the MaxPool operation                      | uint32_t   | {9}                                                   | True     |
 * | layout          | The data layout of the data in DST                                       | DataLayout | TILE or ROW_MAJOR                                     | False    |
 * | ITERATIONS      | The number of iterations to perform (unused)                             | int        | 1 to 8                                                | False    |
 */
// clang-format on
template <int num_rows = 9, ckernel::DataLayout layout = ckernel::DataLayout::TILE, int ITERATIONS = 8>
ALWI void max_reduce_with_indices(uint32_t idst, uint32_t idst_idx) {
    static_assert(num_rows <= 32, "num_rows must be <= 32");
    MATH((llk_math_eltwise_binary_sfpu_max_pool_with_indices<true, DST_ACCUM_MODE, num_rows, ITERATIONS, layout>(
        idst, idst_idx)));
}

/**
 * Please refer to documentation for any_init.
 */
template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
ALWI void max_reduce_with_indices_init() {
    MATH((llk_math_eltwise_binary_sfpu_max_pool_with_indices_init<true, layout>()));
}

// clang-format off
/**
 * Performs reduce operation (sum or average) on a 32x32 tile, placing output values into the first row.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only 32x32 tile dimensions are supported.
 *  - This kernel is optimized for 32x32 tile dimensions and uses VectorMode::RC_custom for customized reduction
 * Only column-wise reduction is supported at this time.
 *
 * | Argument        | Description                                                              | Type      | Valid Range                                           | Required |
 * |-----------------|--------------------------------------------------------------------------|-----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register containing the data to be reduced  | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | pool_type       | The type of reduction operation, SUM or AVG (MAX not supported)          | PoolType  | SUM, AVG                                              | True     |
 * | format          | The data format for the reduction operation                              | DataFormat| Float32, Int32, UInt32                                | True     |
 * | reduce_dim      | The reduction dimension, set to column for column reduce                 | ReduceDim | REDUCE_COL                                            | False    |
 */
// clang-format on
template <PoolType pool_type, DataFormat format, ReduceDim reduce_dim = ReduceDim::REDUCE_COL>
ALWI void sfpu_reduce(uint32_t idst) {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32 ||
            format == DataFormat::UInt16 || format == DataFormat::Float16_b,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32, UInt16, Float16_b");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool type. Supported pool types: SUM, AVG, MAX, MIN");

    // This kernel is optimized for 32x32 tiles and uses RC_custom vector mode for custom reduction
    MATH((llk_math_eltwise_unary_sfpu_reduce<true, pool_type, reduce_dim, format>(idst, VectorMode::RC_custom)));
}

/**
 * @brief Initialization for SFPU reduce kernel.
 *        Must be called before sfpu_reduce() to set up the necessary configurations for reduction operations.
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

    MATH((llk_math_eltwise_unary_sfpu_reduce_init<true, pool_type, format>()));
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

    MATH((llk_math_eltwise_binary_sfpu_add_top_row<format>(dst_tile_0, dst_tile_1, dst_tile_out)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sfpu_add_top_row_init() { MATH((llk_math_eltwise_binary_sfpu_add_top_row_init())); }

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
    MATH((llk_math_eltwise_unary_sfpu_unary_max_int32<APPROX>(idst, param0)));
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
    MATH((llk_math_eltwise_unary_sfpu_unary_max<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_max_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_max_init<APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_alt_complex_rotate90<APPROX>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void alt_complex_rotate90_tile_init() { MATH((llk_math_eltwise_unary_sfpu_alt_complex_rotate90_init<APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_unary_min_int32<APPROX>(idst, param0)));
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
    MATH((llk_math_eltwise_unary_sfpu_unary_min<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_min_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_min_init<APPROX>())); }

ALWI uint32_t get_compute_special_value_flags() {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_get_compute_special_value_flags()));
    return ret_val;
}

ALWI uint32_t get_compute_special_value_flags_fpu(uint32_t special_value_flags_reg) {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_get_compute_special_value_flags_fpu(special_value_flags_reg)));
    return ret_val;
}

ALWI uint32_t get_compute_special_value_flags_sfpu(uint32_t special_value_flags_reg) {
    uint32_t ret_val = 0;
    MATH((ret_val = llk_math_get_compute_special_value_flags_sfpu(special_value_flags_reg)));
    return ret_val;
}

ALWI void clear_compute_special_value_flags() { MATH((llk_math_clear_compute_special_value_flags())); }

ALWI void store_compute_special_value_flags_to_l1(uint32_t l1_addr) {
    MATH((llk_math_store_compute_special_value_flags_to_l1(l1_addr)));
}

}  // namespace ckernel

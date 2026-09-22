// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "ckernel_sfpu_rounding_ops.h"
#else
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 *
 * Each rounding op has its own init. On Blackhole with bf16 DEST, floor/ceil/trunc/round run SFPLOADMACRO-based
 * fast kernels whose SFPU state (LREG constants, macro templates/sequences, replay slots) is programmed by the
 * op-specific init and is different for every op, so the matching <op>_tile_init() MUST be called before
 * <op>_tile(); the inits are not interchangeable.
 */
ALWI void floor_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT_FN(floor, sfpu::_init_floor_, (DST_ACCUM_MODE)));
#else
    MATH(SFPU_UNARY_INIT(unused));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void ceil_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT_FN(ceil, sfpu::_init_ceil_, (DST_ACCUM_MODE)));
#else
    MATH(SFPU_UNARY_INIT(unused));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void trunc_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT_FN(trunc, sfpu::_init_trunc_, (DST_ACCUM_MODE)));
#else
    MATH(SFPU_UNARY_INIT(unused));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void round_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT_FN(round, sfpu::_init_round_, (DST_ACCUM_MODE)));
#else
    MATH(SFPU_UNARY_INIT(unused));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void frac_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void stochastic_round_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

/**
 * Deprecated shared init. It only resets the SFPU counters and does NOT program the per-op fast-kernel state that
 * floor_tile/ceil_tile/trunc_tile/round_tile rely on (Blackhole, bf16 DEST), so using it before those ops yields
 * wrong results. Use the op-specific <op>_tile_init() instead.
 */
// REMOVED: the Blackhole floor/ceil/trunc/round kernels now each program op-specific SFPU state in their own
// init, and those states are mutually exclusive, so one shared init can no longer be correct for all of them.
// Calling this is a hard compile error (a silent fallback would compute garbage on Blackhole); use
// floor_tile_init / ceil_tile_init / trunc_tile_init / round_tile_init / frac_tile_init / stochastic_round_tile_init.
template <typename Removed = void>
ALWI void rounding_op_tile_init() {
    static_assert(
        !std::is_same_v<Removed, void>,
        "rounding_op_tile_init() was removed: call the op-specific init (floor_tile_init, ceil_tile_init, "
        "trunc_tile_init, round_tile_init, frac_tile_init, stochastic_round_tile_init) immediately before the op.");
}

// clang-format off
/**
 * Performs element-wise ceil computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform ceil operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void ceil_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_ceil_,
        (APPROX, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_ceil_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
#endif
}

// clang-format off
/**
 * Performs element-wise floor computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform floor operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void floor_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_floor_,
        (APPROX, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_floor_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
#endif
}

// clang-format off
/**
 * Performs element-wise trunc computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_trunc_,
        (APPROX, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_trunc_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
#endif
}

// clang-format off
/**
 * Performs element-wise computation of the round operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | decimals        | The number of decimal places to round to.                                  | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void round_tile(uint32_t idst, int32_t decimals) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_round_,
        (APPROX, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        decimals));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_round_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, decimals));
#endif
}

// clang-format off
/**
 * Performs element-wise stochastic rounding operation of FP32 values to BF16
 * on each element of a tile in DST register at index tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Note: this operation uses the SFPSTOCHRND instruction; see the known issues on Wormhole and Blackhole:
 * https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
 * https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
 *
 * Return value: None
 *
 * | Argument        | Description                                                                         | Type     | Valid Range                                           | Required |
 * |-----------------|-------------------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the stochastic round on     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void stochastic_round_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_stochastic_round_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
}

// clang-format off
/**
 * Performs element-wise frac computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform frac operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void frac_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_frac_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
}

}  // namespace ckernel

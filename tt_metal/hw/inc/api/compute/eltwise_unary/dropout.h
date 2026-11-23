// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_dropout.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_rand_keyed.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise dropout on each element of a of a tile in DST register at index tile_index.
 * That is each element may be zeroed out based on a given probability or it may be scaled by a given
 * scale factor. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | probability     | A non-negative integer value representing dropout probability              | uint32_t | 0 to INT_MAX (float_probability * (double) INT_MAX)   | True     |
 * | scale_factor    | uint bitwise representation of 32 bit floating point scale factor          | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void dropout_tile(uint32_t idst, uint32_t probability, uint32_t scale_factor) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_dropout, (APPROX), idst, VectorMode::RC, probability, scale_factor));
}

/**
 * This init should be called once in kernel
 */
ALWI void dropout_kernel_init(uint32_t seed = 0) {
    MATH(SFPU_UNARY_INIT_FN_ARGS(dropout, sfpu::dropout_init, (APPROX), seed));
}

// clang-format off
/**
 * Counter-based variant of dropout_tile: zeroes each element of the tile in DST register at index
 * idst, or scales it by scale_factor, based on a draw that is a pure function of (seed, tile_index,
 * position in the tile). The mask does not depend on which core computes the tile, on what ran
 * before, or on the SFPU PRNG state, so re-running with the same seed and tile indices rebuilds it
 * exactly, and Wormhole and Blackhole give identical masks. With the same seed and tile_index, an
 * element is dropped exactly when rand_keyed_tile with from = 0 and scale = 1 gives it a value below
 * probability / 2^31. Call dropout_keyed_tile_init() once before the first call. On Quasar this
 * reseeds the SFPU PRNG per tile and calls dropout_tile, so masks are deterministic per tile but
 * differ from Wormhole and Blackhole. On Wormhole, this operation records replay slots 0-15,
 * programs LREG12-LREG14 and overwrites ADDR_MOD_6 (dest incr 2). Callers sharing that state with
 * other SFPU operations must re-run their inits after it.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | seed            | Seed of the stream                                                         | uint32_t | Any value                                             | True     |
 * | tile_index      | Index of the tile within the stream                                        | uint32_t | 0 to 2^22 - 1 for distinct masks                      | True     |
 * | probability     | A non-negative integer value representing dropout probability              | uint32_t | 0 to INT_MAX (float_probability * (double) INT_MAX)   | True     |
 * | scale_factor    | uint bitwise representation of 32 bit floating point scale factor          | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void dropout_keyed_tile(
    uint32_t idst, uint32_t seed, uint32_t tile_index, uint32_t probability, uint32_t scale_factor) {
#ifdef ARCH_QUASAR
    constexpr uint32_t stream_seed_multiplier = 0x9E3779B9U;
    dropout_kernel_init(seed + tile_index * stream_seed_multiplier);
    dropout_tile(idst, probability, scale_factor);
#else
    MATH(uint32_t face_index = tile_index * 4);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        dropout_keyed,
        (APPROX),
        idst,
        VectorMode::RC,
        probability,
        scale_factor,
        sfpu::keyed_rand_key(seed),
        face_index));
#endif
}

/**
 * Initializes the SFPU for dropout_keyed_tile. Unlike dropout_kernel_init it takes no seed and does
 * not touch the SFPU PRNG.
 */
ALWI void dropout_keyed_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(dropout));
#endif
}

}  // namespace ckernel

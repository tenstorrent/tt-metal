// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_instr_params.h"
#include "ckernel_sfpu_rand.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_rand_keyed.h"
#endif
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise rand on each element of a of a tile in DST register at index tile_index.
 * That is each element is overwritten with a randomly generated float.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 * This operation records replay slots 0-15 when scale is finite and at least
 * 2^-95, and slots 0-16 otherwise. Callers sharing the replay buffer with other
 * SFPU operations must reinitialize it as needed.
 * On Wormhole, this operation also programs LREG12 and LREG13. Callers must
 * restore those programmable constants before another SFPU operation relies on them.
 * This operation overwrites ADDR_MOD_6 (dest incr 2) on every call; that is the only
 * state it re-establishes, so a rand_tile with no rand_tile_init in the same kernel
 * (keeping the PRNG stream seeded by an earlier rand_tile_init or a host-set seed) is
 * valid only while the SFPU control register still holds the state the generic SFPU
 * init leaves (control register cleared). An init that changes SFPU control state
 * (e.g. topk_tile_init, which enables index tracking) must be followed by a clear before
 * rand_tile, and the caller has to issue it: any generic SFPU *_tile_init clears it, or
 * _init_sfpu_config_reg() directly; the topk ops themselves leave index tracking on at
 * exit. rand_tile_init also clears it, but reseeds the PRNG stream, so it is not the
 * tool for the seedless case above. Ops relying on ADDR_MOD_6 from
 * their own init (e.g. topk_tile_init, square, sfpu reduce, matmul with THROTTLE_LEVEL)
 * must re-run that init after rand_tile.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                           | Required  |
 * |----------------|---------------------------------------------------------------|----------|-------------------------------------------------------|-----------|
 * | tile_index     | The index of the tile in the DST register buffer              | uint32_t | Must be less than the size of the DST register buffer | True      |
 * | from           | FP32 bit pattern for the inclusive lower bound                | uint32_t | Any supported FP32 value                              | True      |
 * | scale          | FP32 bit pattern producing [from, from + scale], inclusively  | uint32_t | Must be non-negative; zero produces a constant tile   | True      |
 */
// clang-format on
ALWI void rand_tile(uint32_t idst, uint32_t from, uint32_t scale) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rand, (APPROX), idst, VectorMode::RC, from, scale));
}

/**
 * Initializes the random generator with seed + stream_id * 0x9E3779B9 modulo
 * 2^32. Distinct stream IDs domain-separate related deterministic work ranges;
 * stream_id=0 leaves the seed unchanged.
 */
ALWI void rand_tile_init(uint32_t seed = 0, uint32_t stream_id = 0) {
    constexpr uint32_t stream_seed_multiplier = 0x9E3779B9U;
    MATH(SFPU_UNARY_INIT_FN_ARGS(unused, sfpu::rand_init, (APPROX), seed + stream_id * stream_seed_multiplier));
}

// clang-format off
/**
 * Counter-based variant of rand_tile: overwrites each element of the tile in DST register at index
 * idst with a value in [from, from + scale] that is a pure function of (seed, tile_index, position
 * in the tile). The result does not depend on which core computes the tile, on what ran before, or
 * on the SFPU PRNG state, and Wormhole and Blackhole give identical values. Give every tile of a
 * stream its own tile_index (for example its index in the tensor); indices are unique for the first
 * 2^22 tiles. Call rand_keyed_tile_init() once before the first call. No seeding or wait is needed.
 * On Quasar this reseeds the SFPU PRNG with rand_tile_init(seed, tile_index) and calls rand_tile, so
 * values are deterministic per tile but differ from Wormhole and Blackhole.
 * On Wormhole, this operation records replay slots 0-15, programs LREG12-LREG14 and overwrites
 * ADDR_MOD_6 (dest incr 2). Callers sharing that state with other SFPU operations must re-run their
 * inits after it.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                           | Required  |
 * |----------------|---------------------------------------------------------------|----------|-------------------------------------------------------|-----------|
 * | idst           | The index of the tile in the DST register buffer              | uint32_t | Must be less than the size of the DST register buffer | True      |
 * | seed           | Seed of the stream                                            | uint32_t | Any value                                             | True      |
 * | tile_index     | Index of the tile within the stream                           | uint32_t | 0 to 2^22 - 1 for distinct values                     | True      |
 * | from           | FP32 bit pattern for the inclusive lower bound                | uint32_t | Any supported FP32 value                              | True      |
 * | scale          | FP32 bit pattern producing [from, from + scale], inclusively  | uint32_t | Must be non-negative; zero produces a constant tile   | True      |
 */
// clang-format on
ALWI void rand_keyed_tile(uint32_t idst, uint32_t seed, uint32_t tile_index, uint32_t from, uint32_t scale) {
#ifdef ARCH_QUASAR
    rand_tile_init(seed, tile_index);
    rand_tile(idst, from, scale);
#else
    MATH(uint32_t face_index = tile_index * 4);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        rand_keyed,
        (APPROX),
        idst,
        VectorMode::RC,
        from,
        scale,
        sfpu::keyed_rand_key(seed),
        face_index));
#endif
}

/**
 * Initializes the SFPU for rand_keyed_tile. Unlike rand_tile_init it takes no seed and does not touch
 * the SFPU PRNG.
 */
ALWI void rand_keyed_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(unused));
#endif
}

}  // namespace ckernel

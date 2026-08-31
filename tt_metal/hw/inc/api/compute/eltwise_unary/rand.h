// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_instr_params.h"
#include "ckernel_sfpu_rand.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
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

}  // namespace ckernel

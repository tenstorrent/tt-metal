// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// The two-element body is recorded into replay slots [0, TILED_PROD_REPLAY_LEN) on every call.
constexpr std::uint32_t TILED_PROD_REPLAY_SLOT = 0;
constexpr int TILED_PROD_ELEMENTS_PER_REPLAY = 2;
// SFPLOAD, SFPSTORE, SFPMUL per element; the window must match the body sfpi emits.
constexpr std::uint32_t TILED_PROD_INSTRUCTIONS_PER_ELEMENT = 3;
constexpr std::uint32_t TILED_PROD_REPLAY_LEN =
    TILED_PROD_INSTRUCTIONS_PER_ELEMENT * static_cast<std::uint32_t>(TILED_PROD_ELEMENTS_PER_REPLAY);
// Elements per lane in one tile.
constexpr int TILED_PROD_STEPS = static_cast<int>(TILE_NUM_FACES * FACE_R_DIM / SFP_DESTREG_STRIDE);
// The recording does not execute, so the tail stores the last two elements.
constexpr int TILED_PROD_TAIL_STORES = 2;
constexpr int TILED_PROD_REPLAYS = (TILED_PROD_STEPS - TILED_PROD_TAIL_STORES) / TILED_PROD_ELEMENTS_PER_REPLAY;
static_assert(
    TILED_PROD_REPLAYS * TILED_PROD_ELEMENTS_PER_REPLAY + TILED_PROD_TAIL_STORES == TILED_PROD_STEPS,
    "calculate_tiled_prod: the replays and the tail must store every element once");

/**
 * @brief Programs what @ref calculate_tiled_prod reads: ADDR_MOD_6 = {dest.incr = SFP_DESTREG_STRIDE}
 * for its stores and 1.0 in vConstFloatPrgm0 for its first multiply.
 *
 * @note Both are shared: call this after any other SFPU init.
 */
inline void tiled_prod_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
    vConstFloatPrgm0 = 1.0f;
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Running product along each SFPU lane through faces 0-3 of one Dest tile, stored in place.
 *
 * @tparam APPROXIMATION_MODE: Unused; kept for the SFPU dispatch signature.
 * @tparam ITERATIONS: Steps per face; must span exactly one face.
 * @note Run once per tile under VectorMode::RC_custom: the product is carried from face to face.
 * @note Call @ref tiled_prod_init first. The body overwrites replay slots [0, TILED_PROD_REPLAY_LEN).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    static_assert(
        ITERATIONS * static_cast<int>(SFP_DESTREG_STRIDE) == static_cast<int>(FACE_R_DIM),
        "calculate_tiled_prod: ITERATIONS steps of SFP_DESTREG_STRIDE Dest rows must cover one face");
    constexpr int STEPS = static_cast<int>(TILE_NUM_FACES) * ITERATIONS;  // 32 elements per lane
    static_assert(STEPS == TILED_PROD_STEPS, "calculate_tiled_prod: one call covers one tile");
    // Per element: LOAD the next v, STORE the product (ADDR_MOD_6 steps Dest), MUL. The load fills the
    // MUL -> STORE latency slot.
    //
    // The first multiply reads 1.0 from vConstFloatPrgm0, not a literal: sfpi folds `1.0f * v` to `v`,
    // which would store a -0.0 first element as -0.0 where SFPMUL gives +0.0.
    //
    // The empty asm statements stop sfpi's own replay recording from spanning lltt::record when calls
    // are back to back; a nested recording hangs the core.
    asm volatile("" ::: "memory");
    vFloat v = dst_reg[0];
    vFloat result = vConstFloatPrgm0 * v;
    asm volatile("" ::: "memory");
    TTI_REPLAY(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN, 0, 0, 0, 1);
#pragma GCC unroll TILED_PROD_ELEMENTS_PER_REPLAY
    for (int k = 0; k < TILED_PROD_ELEMENTS_PER_REPLAY; k++) {
        v = dst_reg[1];
        dst_reg[0].mode(ADDR_MOD_6) = result;
        result *= v;
    }
#pragma GCC unroll TILED_PROD_REPLAYS
    for (int i = 0; i < TILED_PROD_REPLAYS; i++) {
        TTI_REPLAY(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN, 0, 0, 0, 0);
    }
    v = dst_reg[1];
    dst_reg[0].mode(ADDR_MOD_6) = result;
    result *= v;
    dst_reg[0].mode(ADDR_MOD_6) = result;
    asm volatile("" ::: "memory");
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Replay-buffer slot and length of the two-element body recorded by every calculate_tiled_prod
// call. Slot 0 is where sfpi's own loop rolling put the previous body, so no other kernel's
// recording is displaced that was not already.
constexpr std::uint32_t TILED_PROD_REPLAY_SLOT = 0;
constexpr std::uint32_t TILED_PROD_REPLAY_LEN = 6;

inline void tiled_prod_init() {
    // The stores below advance the Dest counter by one SFPU row (SFP_DESTREG_STRIDE) each, under the
    // SFPU addr_mod base (real slot ADDR_MOD_6, selected as ADDR_MOD_2 from the instruction), as the
    // comparison and unary max/min kernels do; the loads keep ADDR_MOD_3 (real ADDR_MOD_7, no increment).
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
/**
 * @brief Running product along each SFPU lane through faces 0-3 of one Dest tile, each partial
 * product stored in place.
 *
 * @tparam APPROXIMATION_MODE: Unused; kept for the SFPU dispatch signature.
 * @tparam ITERATIONS: Steps per face (two Dest rows each); must span exactly one face, enforced
 * by the static_assert below.
 * @note Run once per tile under VectorMode::RC_custom, not once per face. The product is carried
 * in a register from face to face, so every store lands inside the tile being processed; a
 * per-face call with a ninth, carry-over store would write the first rows of the next Dest tile,
 * which belongs to another tile of the batch or to the packer's half.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    // One call must cover exactly one tile: each step advances dst_reg by SFP_DESTREG_STRIDE Dest
    // rows, so ITERATIONS steps have to span one face. Fewer leave part of the tile untouched,
    // more walk into the next Dest tile.
    static_assert(
        ITERATIONS * static_cast<int>(SFP_DESTREG_STRIDE) == static_cast<int>(FACE_R_DIM),
        "calculate_tiled_prod: ITERATIONS steps of SFP_DESTREG_STRIDE Dest rows must cover one face");
    // Per element the chain is LOAD the next v ; STORE the product ; MUL (product *= v). The load
    // sits in the MUL -> STORE latency slot of the previous element, so sfpi needs no SFPNOP except
    // before the last store, and the store advances the Dest counter (ADDR_MOD_2, see
    // tiled_prod_init) instead of a TTINCRWC. Two elements of that body are recorded once and
    // replayed; the product is their only carried value, and sfpi multiplies it in place
    // (`sfpmul L0, L0, L1`), so each replay picks it up where the previous one left it.
    //
    // TILED_PROD_REPLAY_LEN is the instruction count sfpi emits for the recorded loop below:
    //   2x (SFPLOAD, SFPSTORE, SFPMUL)
    // If the body changes, re-check it against the generated kernel.
    //
    // The product is seeded with 1.0 through a pinned LReg rather than a literal: sfpi folds a
    // literal `1.0f * v` to `v`, which skips the first SFPMUL and stores a -0.0 first element
    // as -0.0 instead of the +0 SFPMUL returns. This keeps the previous loop's numerics
    // (`result = 1.0f; result *= v`): the same multiplies in the same order.
    constexpr int STEPS = static_cast<int>(TILE_NUM_FACES) * ITERATIONS;  // 32 elements per lane
    l_reg[LRegs::LReg0] = vFloat(1.0f);
    vFloat result = l_reg[LRegs::LReg0];
    vFloat v = dst_reg[0];
    result *= v;
    lltt::record<lltt::Exec>(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
#pragma GCC unroll 2
    for (int k = 0; k < 2; k++) {
        v = dst_reg[1];
        dst_reg[0].mode(ADDR_MOD_2) = result;
        result *= v;
    }
    // The recording ran elements 1 and 2 and the last element runs below, so the replays cover
    // elements 3 to STEPS - 2.
#pragma GCC unroll 16
    for (int i = 0; i < STEPS / 2 - 2; i++) {
        lltt::replay(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
    }
    v = dst_reg[1];
    dst_reg[0].mode(ADDR_MOD_2) = result;
    result *= v;
    dst_reg[0].mode(ADDR_MOD_2) = result;
}

}  // namespace sfpu
}  // namespace ckernel

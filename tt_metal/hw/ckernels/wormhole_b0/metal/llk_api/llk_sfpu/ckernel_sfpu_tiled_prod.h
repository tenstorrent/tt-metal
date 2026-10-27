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
    // Per element the chain is MUL (product *= v) ; LOAD the next v ; STORE the product. The load
    // of element d+1 sits in the MUL -> STORE latency slot, so no SFPNOP is needed except for the
    // last element; the loaded value alternates between LREG1 and LREG2 so a two-element body can
    // be recorded once and replayed. SFPMUL with LCONST_0 as the addend is a plain multiply, and
    // the product starts at 1.0 (LCONST_1), as the previous sfpi loop did (`result = 1.0f; result
    // *= v`), so the instruction stream computes the same products in the same order.
    constexpr int STEPS = static_cast<int>(TILE_NUM_FACES) * ITERATIONS;  // 32 elements per lane
    TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    lltt::record<lltt::Exec>(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, SFP_DESTREG_STRIDE);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, SFP_DESTREG_STRIDE);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
#pragma GCC unroll 16
    for (int i = 0; i < STEPS / 2 - 2; i++) {
        lltt::replay(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
    }
    // Last pair: nothing to prefetch past the tile, so the final MUL -> STORE gap is an SFPNOP.
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, SFP_DESTREG_STRIDE);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
}

}  // namespace sfpu
}  // namespace ckernel

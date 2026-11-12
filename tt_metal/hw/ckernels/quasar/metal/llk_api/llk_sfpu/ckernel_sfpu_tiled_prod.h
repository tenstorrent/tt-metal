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

// Replay-buffer window of the two-element body that every calculate_tiled_prod call records and
// then replays. The recording is made at the start of each call, so the kernel depends on no
// earlier recording, and it overwrites slots [0, TILED_PROD_REPLAY_LEN) of replay bank 0 (the bank
// mac, gcd and binary max/min record into). Kernels that record in their init and replay later
// record again when their init runs, which the compute API requires after any other op.
constexpr std::uint32_t TILED_PROD_REPLAY_SLOT = 0;
// Elements per recorded window: each element stores one partial product.
constexpr int TILED_PROD_ELEMENTS_PER_REPLAY = 2;
// Instructions sfpi emits per element of the recorded loop: SFPLOAD, SFPSTORE, SFPMUL. A window
// that does not match the emitted body replays the wrong instructions and corrupts the tile.
constexpr std::uint32_t TILED_PROD_INSTRUCTIONS_PER_ELEMENT = 3;
constexpr std::uint32_t TILED_PROD_REPLAY_LEN =
    TILED_PROD_INSTRUCTIONS_PER_ELEMENT * static_cast<std::uint32_t>(TILED_PROD_ELEMENTS_PER_REPLAY);
// Elements per lane in one tile: SFP_DESTREG_STRIDE Dest rows per element, FACE_R_DIM rows per face.
constexpr int TILED_PROD_STEPS = static_cast<int>(TILE_NUM_FACES * FACE_R_DIM / SFP_DESTREG_STRIDE);
// Stores per lane: Quasar's REPLAY cannot execute while loading, so the recording stores nothing; the
// tail stores the last TILED_PROD_TAIL_STORES elements (the one whose load it issues and the final
// product) and the replays store every element before them.
constexpr int TILED_PROD_TAIL_STORES = 2;
constexpr int TILED_PROD_REPLAYS = (TILED_PROD_STEPS - TILED_PROD_TAIL_STORES) / TILED_PROD_ELEMENTS_PER_REPLAY;
static_assert(
    TILED_PROD_REPLAYS * TILED_PROD_ELEMENTS_PER_REPLAY + TILED_PROD_TAIL_STORES == TILED_PROD_STEPS,
    "calculate_tiled_prod: the replays and the tail must store every element once");

inline void tiled_prod_init() {
    // ADDR_MOD_6: dest += SFP_DESTREG_STRIDE, one SFPU row. calculate_tiled_prod's SFPSTOREs name it
    // directly (the SFPU load/store addr_mod field is 3 bits wide on Quasar), so each store advances the
    // Dest counter; its SFPLOADs keep ADDR_MOD_7 (no increment). ADDR_MOD_6 is not reserved: each SFPU
    // op that advances Dest through it (square, typecast, comparison, ...) programs its own increment
    // in its init, which the unary init dispatch runs on every op switch.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
    // The product's 1.0 seed, read by the first SFPMUL (see calculate_tiled_prod).
    vConstFloatPrgm0 = 1.0f;
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
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
 * @note Call @ref tiled_prod_init after any other SFPU init: the stores advance Dest through
 * ADDR_MOD_6, the first multiply reads 1.0 from vConstFloatPrgm0, and the body overwrites replay
 * bank 0 slots [0, TILED_PROD_REPLAY_LEN).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    // One call must cover exactly one tile: each step advances dst_reg by SFP_DESTREG_STRIDE Dest
    // rows, so ITERATIONS steps have to span one face. Fewer leave part of the tile untouched,
    // more walk into the next Dest tile.
    static_assert(
        ITERATIONS * static_cast<int>(SFP_DESTREG_STRIDE) == static_cast<int>(FACE_R_DIM),
        "calculate_tiled_prod: ITERATIONS steps of SFP_DESTREG_STRIDE Dest rows must cover one face");
    constexpr int STEPS = static_cast<int>(TILE_NUM_FACES) * ITERATIONS;  // 32 elements per lane
    static_assert(STEPS == TILED_PROD_STEPS, "calculate_tiled_prod: one call covers one tile");
    // Per element the chain is LOAD the next v ; STORE the product ; MUL (product *= v). The load
    // sits in the MUL -> STORE latency slot of the previous element, and the store advances the
    // Dest counter (ADDR_MOD_6, see tiled_prod_init) instead of a TTINCRWC. Two elements of that
    // body are recorded once (without executing: Quasar's REPLAY cannot execute while loading) and
    // replayed; the product is their only carried value, and sfpi multiplies it in place, so each
    // replay picks it up where the previous one left it.
    //
    // The first element is multiplied by 1.0 from vConstFloatPrgm0 (set in tiled_prod_init) rather
    // than by a literal: sfpi folds `1.0f * v` to `v`, which skips the first SFPMUL and stores a
    // -0.0 first element as -0.0 instead of the +0 SFPMUL returns. Reading the constant register
    // keeps every element's SFPMUL, as the plain loop did, and costs no instruction per call.
    //
    // sfpi records repeated instruction sequences into the replay buffer on its own and treats
    // lltt::record as one more instruction, so with back-to-back calls (an unrolled tile loop) it
    // can record a window that contains this kernel's record and nest one recording in another,
    // which hangs the core. The empty asm statements (no instruction is emitted) keep its windows
    // out of the head, the recorded body and the tail.
    asm volatile("" ::: "memory");
    vFloat v = dst_reg[0];
    vFloat result = vConstFloatPrgm0 * v;
    asm volatile("" ::: "memory");
    lltt::record<lltt::NoExec>(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
#pragma GCC unroll TILED_PROD_ELEMENTS_PER_REPLAY
    for (int k = 0; k < TILED_PROD_ELEMENTS_PER_REPLAY; k++) {
        v = dst_reg[1];
        dst_reg[0].mode(ADDR_MOD_6) = result;
        result *= v;
    }
    // The replays store every element before the tail's (TILED_PROD_TAIL_STORES).
#pragma GCC unroll TILED_PROD_REPLAYS
    for (int i = 0; i < TILED_PROD_REPLAYS; i++) {
        lltt::replay(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
    }
    v = dst_reg[1];
    dst_reg[0].mode(ADDR_MOD_6) = result;
    result *= v;
    dst_reg[0].mode(ADDR_MOD_6) = result;
    asm volatile("" ::: "memory");
}

}  // namespace sfpu
}  // namespace ckernel

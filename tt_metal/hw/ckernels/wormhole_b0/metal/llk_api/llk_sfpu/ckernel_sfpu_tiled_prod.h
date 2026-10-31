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

// Replay-buffer window of the two-element body that every calculate_tiled_prod call records and
// then replays. The recording is made at the start of each call, so the kernel depends on no
// earlier recording, and it overwrites slots [0, TILED_PROD_REPLAY_LEN) of the SFPU half of the
// replay buffer. Kernels that record in their init and replay later (mac, cumsum, cumprod) record
// again when their init runs, which the compute API requires after any other op.
constexpr std::uint32_t TILED_PROD_REPLAY_SLOT = 0;
// Elements per recorded window: each element stores one partial product.
constexpr int TILED_PROD_ELEMENTS_PER_REPLAY = 2;
// Instructions sfpi emits per element of the recorded loop: SFPLOAD, SFPSTORE, SFPMUL. A window
// that does not match the emitted body replays the wrong instructions and corrupts the tile,
// which the full-tile bit-exact check in tt-llk's test_sfpu_tiled_prod.py catches.
constexpr std::uint32_t TILED_PROD_INSTRUCTIONS_PER_ELEMENT = 3;
constexpr std::uint32_t TILED_PROD_REPLAY_LEN =
    TILED_PROD_INSTRUCTIONS_PER_ELEMENT * static_cast<std::uint32_t>(TILED_PROD_ELEMENTS_PER_REPLAY);
static_assert(
    TILED_PROD_REPLAY_SLOT + TILED_PROD_REPLAY_LEN <= ckernel::math::replay_buf_offset,
    "calculate_tiled_prod: the recorded window must stay in the SFPU half of the replay buffer");
// Elements per lane in one tile: SFP_DESTREG_STRIDE Dest rows per element, FACE_R_DIM rows per face.
constexpr int TILED_PROD_STEPS = static_cast<int>(TILE_NUM_FACES * FACE_R_DIM / SFP_DESTREG_STRIDE);
// Stores per lane: the recording stores the first TILED_PROD_ELEMENTS_PER_REPLAY elements and the
// tail the last TILED_PROD_TAIL_STORES (the one whose load it issues and the final product), so the
// replays store the elements between them.
constexpr int TILED_PROD_TAIL_STORES = 2;
constexpr int TILED_PROD_REPLAYS =
    (TILED_PROD_STEPS - TILED_PROD_ELEMENTS_PER_REPLAY - TILED_PROD_TAIL_STORES) / TILED_PROD_ELEMENTS_PER_REPLAY;
static_assert(
    (TILED_PROD_REPLAYS + 1) * TILED_PROD_ELEMENTS_PER_REPLAY + TILED_PROD_TAIL_STORES == TILED_PROD_STEPS,
    "calculate_tiled_prod: the recording, the replays and the tail must store every element once");

#if defined(MM_THROTTLE) && MM_THROTTLE > 0
// A throttled matmul (MM_THROTTLE > 0) programs ADDR_MOD_6 itself in matmul_init, which fused
// matmul kernels run between the activation's init and its call, so there the kernel walks Dest
// with dst_reg++ and leaves ADDR_MOD_6 alone.
constexpr bool TILED_PROD_STORE_ADVANCES_DEST = false;
#else
constexpr bool TILED_PROD_STORE_ADVANCES_DEST = true;
#endif

inline void tiled_prod_init() {
    if constexpr (TILED_PROD_STORE_ADVANCES_DEST) {
        // ADDR_MOD_6: dest += SFP_DESTREG_STRIDE, one SFPU row. calculate_tiled_prod's SFPSTOREs
        // select it as ADDR_MOD_2 (the SFPU call sets the addr_mod base, which maps 0..3 to 4..7),
        // so each store advances the Dest counter; its SFPLOADs keep ADDR_MOD_3 (real ADDR_MOD_7,
        // no increment). ADDR_MOD_6 is not reserved: each SFPU op that advances Dest through it
        // (comparison, unary max/min, typecast, ...) programs its own increment in its init, and
        // llk_math_eltwise_unary_sfpu_init runs that init on every op switch.
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
        // The product's 1.0 seed, read by the first SFPMUL (see calculate_tiled_prod).
        vConstFloatPrgm0 = 1.0f;
    }
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
 * @note Call @ref tiled_prod_init after any other SFPU init: the stores advance Dest through
 * ADDR_MOD_6, the first multiply reads 1.0 from vConstFloatPrgm0, and the body overwrites replay
 * slots [0, TILED_PROD_REPLAY_LEN).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    // One call must cover exactly one tile: each step covers SFP_DESTREG_STRIDE Dest rows, so
    // ITERATIONS steps have to span one face. Fewer leave part of the tile untouched,
    // more walk into the next Dest tile.
    static_assert(
        ITERATIONS * static_cast<int>(SFP_DESTREG_STRIDE) == static_cast<int>(FACE_R_DIM),
        "calculate_tiled_prod: ITERATIONS steps of SFP_DESTREG_STRIDE Dest rows must cover one face");
    constexpr int STEPS = static_cast<int>(TILE_NUM_FACES) * ITERATIONS;  // 32 elements per lane
    static_assert(STEPS == TILED_PROD_STEPS, "calculate_tiled_prod: one call covers one tile");
    if constexpr (!TILED_PROD_STORE_ADVANCES_DEST) {
        vFloat result = 1.0f;
#pragma GCC unroll 8
        for (int d = 0; d < STEPS; d++) {
            vFloat v = dst_reg[0];
            result *= v;
            dst_reg[0] = result;
            dst_reg++;
        }
        return;
    }
    // Per element the chain is LOAD the next v ; STORE the product ; MUL (product *= v). The load
    // sits in the MUL -> STORE latency slot of the previous element, so sfpi needs no SFPNOP except
    // before the last store, and the store advances the Dest counter (ADDR_MOD_2, see
    // tiled_prod_init) instead of a TTINCRWC. Two elements of that body are recorded once and
    // replayed; the product is their only carried value, and sfpi multiplies it in place
    // (`sfpmul L0, L0, L1`), so each replay picks it up where the previous one left it.
    //
    // The first element is multiplied by 1.0 from vConstFloatPrgm0 (set in tiled_prod_init) rather
    // than by a literal: sfpi folds `1.0f * v` to `v`, which skips the first SFPMUL and stores a
    // -0.0 first element as -0.0 instead of the +0 SFPMUL returns. Reading the constant register
    // keeps every element's SFPMUL, as the plain loop above does, and costs no instruction per call.
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
    lltt::record<lltt::Exec>(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
#pragma GCC unroll TILED_PROD_ELEMENTS_PER_REPLAY
    for (int k = 0; k < TILED_PROD_ELEMENTS_PER_REPLAY; k++) {
        v = dst_reg[1];
        dst_reg[0].mode(ADDR_MOD_2) = result;
        result *= v;
    }
    // The replays store the elements between the recording's and the tail's (TILED_PROD_TAIL_STORES).
#pragma GCC unroll TILED_PROD_REPLAYS
    for (int i = 0; i < TILED_PROD_REPLAYS; i++) {
        lltt::replay(TILED_PROD_REPLAY_SLOT, TILED_PROD_REPLAY_LEN);
    }
    v = dst_reg[1];
    dst_reg[0].mode(ADDR_MOD_2) = result;
    result *= v;
    dst_reg[0].mode(ADDR_MOD_2) = result;
    asm volatile("" ::: "memory");
}

}  // namespace sfpu
}  // namespace ckernel

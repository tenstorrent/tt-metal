// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Replay window of the calculate_abs_int32 row body, in replay bank 0 (the bank mac, gcd and binary
// max/min record into, and the one sfpi rolls loop bodies into).
constexpr std::uint32_t ABS_INT32_REPLAY_SLOT = 0;
constexpr std::uint32_t ABS_INT32_REPLAY_LEN = 3;

/**
 * @brief Programs the state calculate_abs_int32 reads: ADDR_MOD_6 = {dest.incr = SFP_ROWS}, which its
 *        store advances Dest through (the SFPU load/store addr_mod field is 3 bits wide on Quasar, so
 *        the store names the slot directly), and the row body in the replay window. Float abs reads
 *        neither.
 * @note Run it after any other SFPU op's init and before calculate_abs_int32, and again after anything
 *       that records into the replay window: another op's init, or an sfpi row loop, which sfpi rolls
 *       into the replay buffer from slot 0.
 */
inline void abs_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // sfpi emits the row body (SFPLOAD, SFPABS, SFPSTORE: sfpi::abs(vInt) is the dedicated integer
    // SFPABS) straight into the replay buffer. Recorded without executing: Quasar's REPLAY cannot
    // execute while loading, and nothing is in Dest yet. The store's ADDR_MOD_6 advance stands in for
    // a TTINCRWC, which would be a fourth instruction in the window.
    lltt::record<lltt::NoExec>(ABS_INT32_REPLAY_SLOT, ABS_INT32_REPLAY_LEN);
    const sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
    sfpi::dst_reg[0].mode<sfpi::DataLayout::M32>(ADDR_MOD_6) = sfpi::abs(v);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

/**
 * @brief Int32 abs for ITERATIONS Dest rows: one REPLAY of the row body abs_init recorded per row.
 * @note Call abs_init first; this only replays the window and advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(ABS_INT32_REPLAY_SLOT, ABS_INT32_REPLAY_LEN);
    }
}

}  // namespace sfpu
}  // namespace ckernel

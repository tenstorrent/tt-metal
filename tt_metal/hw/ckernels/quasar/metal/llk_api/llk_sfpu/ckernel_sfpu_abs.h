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

inline void abs_init() {
    // calculate_abs_int32 advances Dest through ADDR_MOD_6 on its store (the SFPU load/store addr_mod
    // field is 3 bits wide on Quasar, so the store names the slot directly). Float abs does not read it.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
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

// Replay window of the row body: recorded into replay bank 0 on every call (the bank mac, gcd and
// binary max/min record into), so an op that records into those slots at init time must be
// re-initialised after this one. Needs ADDR_MOD_6 = {dest.incr = SFP_ROWS} from abs_init().
constexpr std::uint32_t ABS_INT32_REPLAY_SLOT = 0;
constexpr std::uint32_t ABS_INT32_REPLAY_LEN = 3;

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    static_assert(ITERATIONS >= 1, "the recorded row body is replayed once per row");
    // Raw TTI on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi form was the same
    // three Tensix instructions plus a TTINCRWC per row (an eight-instruction two-row body recorded and
    // replayed by sfpi). Quasar's REPLAY cannot execute while loading, so the three-instruction row body
    // is recorded without executing and then replayed once per row: one RISC issue per row. The store
    // advances Dest by one sfpi row (SFP_ROWS Dest counter rows) through ADDR_MOD_6.
    // sfpmem::INT32 (mode 4) is the raw 32-bit Dest access the sfpi I32/M32 layouts used, so the bytes
    // moved are unchanged; SFPABS mod 0 is the two's-complement abs sfpi::abs(vInt) lowered to.
    lltt::record<lltt::NoExec>(ABS_INT32_REPLAY_SLOT, ABS_INT32_REPLAY_LEN);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
    TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_6, 0 /*done*/, 0 /*dest_reg_addr*/);
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(ABS_INT32_REPLAY_SLOT, ABS_INT32_REPLAY_LEN);
    }
}

}  // namespace sfpu
}  // namespace ckernel

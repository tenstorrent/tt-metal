// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void abs_init() {
    // calculate_abs_int32 advances dest through ADDR_MOD_6 on its store (Blackhole has no addr_mod base,
    // so the store names the physical slot). Float abs does not read it.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
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

// Records the row body into replay slots [0, 3) on every call, so an op that records into those slots at
// init time must be re-initialised after this one. Needs ADDR_MOD_6 = {dest.incr = 2} from abs_init().
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    static_assert(ITERATIONS >= 1, "the row body is recorded while executing the first row");
    constexpr std::uint32_t replay_start = 0;
    constexpr std::uint32_t replay_len = 3;
    static_assert(
        replay_start + replay_len <= ckernel::math::replay_buf_offset,
        "the row body must stay in the SFPU half of the replay buffer");
    // Raw TTI on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi form was the same
    // three Tensix instructions plus a TTINCRWC and a rolled RISC loop (addi/bnez) per row. The row body
    // is recorded (and executed) once and replayed for the remaining rows: one RISC issue per row. The
    // store advances dest by one sfpi row (two dest counter steps) through ADDR_MOD_6.
    // INT32 load/store is the two's-complement layout the sfpi I32 access used; on Blackhole it is the
    // same Dest access as INT32_2S_COMP, so the bytes moved are unchanged.
    lltt::record<lltt::Exec>(replay_start, replay_len);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0 /*dest_reg_addr*/);
    TTI_SFPABS(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0 /*dest_reg_addr*/);
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(replay_start, replay_len);
    }
}

}  // namespace sfpu
}  // namespace ckernel

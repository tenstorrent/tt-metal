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
    // calculate_abs_int32 advances dest through physical addr_mod slot 6 on its store. Unary SFPU sets
    // addr_mod_base, so the store names ADDR_MOD_2, which maps to slot 6. Float abs does not read it.
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
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    // The row body is recorded (and executed) once and replayed for the remaining rows: one RISC issue
    // per row. The store advances dest by one sfpi row (two dest counter steps) through ADDR_MOD_6.
    lltt::record<lltt::Exec>(replay_start, replay_len);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0 /*dest_reg_addr*/);
    TTI_SFPABS(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0 /*dest_reg_addr*/);
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(replay_start, replay_len);
    }
}
}  // namespace sfpu
}  // namespace ckernel

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

inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    // The row body is recorded (and executed) once into replay slots [0,4) -- the SFPU half of the
    // buffer -- and replayed for the remaining rows: the same four Tensix instructions per row,
    // one RISC issue instead of four. TTI_INCRWC(0, 2, 0, 0) is what `dst_reg++` emits.
    lltt::record<lltt::Exec>(0 /*start*/, 4 /*length*/);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0 /*dest_reg_addr*/);
    TTI_SFPABS(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0 /*dest_reg_addr*/);
    TTI_INCRWC(0 /*rwc_cr*/, 2 /*rwc_d*/, 0 /*rwc_b*/, 0 /*rwc_a*/);
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0 /*start*/, 4 /*length*/);
    }
}
}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void heaviside_init() {
    // The row's last SFPSTORE steps dest by the two counter rows one SFPU row covers, so the
    // body needs no TTINCRWC. The unary SFPU addr_mod base maps the body's ADDR_MOD_2 onto
    // physical slot 6 (the typecast / square convention).
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(std::uint32_t value) {
    // heaviside(v) = `value` at v == 0 (either zero), 0.0 for a set sign bit, 1.0 otherwise.
    //
    // Raw SFPU body, 7 instructions per row. The sfpi form (copysgn + add, then an
    // sfpi::abs(v) == 0 arm) costs 11: SFPABS + SFPSETCC for the zero test, two SFPMOVs to
    // merge the arms, and a RISC loop. Here the row is three stores of constants under
    // nested predicates: `value` to every lane; then SFPLZ with NOSGN|CC_NE0 enables the lanes
    // with (bits & 0x7fffffff) != 0 -- the same |v| != 0 test as before, so -0.0 keeps `value`
    // -- which take 1.0; then SFPSETCC LT0 narrows the enable to the lanes whose sign bit is
    // set (SFPSETCC only writes the flags of enabled lanes), which take 0.0. Both forms
    // dispatch on the sign bit rather than a float compare, so NaN lands where it did:
    // a NaN with a clear sign bit is 1.0, with a set one 0.0.
    _sfpu_load_imm32_(p_sfpu::LREG4, value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);     // every lane: value
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);  // |v| != 0
        TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);  //   -> 1.0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);            //   and sign bit set
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);  //   -> 0.0; steps dest
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    }
}

}  // namespace sfpu
}  // namespace ckernel

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

namespace ckernel {
namespace sfpu {

inline void heaviside_init() {
    // The row's last SFPSTORE steps dest by one SFPU row (SFP_DESTREG_STRIDE counter rows, the step
    // sfpi's dst_reg++ takes), so the body needs no TTINCRWC. Same ADDR_MOD_6 value as typecast_init
    // and the comp *_init.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = sfpi::SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Addressing: the body names ADDR_MOD_3 and ADDR_MOD_2, which reach physical slots 7 and 6 because
// _llk_math_eltwise_sfpu_start_ sets the addr_mod base to 4 (math::set_addr_mod_base). Slot 7 is
// {dest.incr = 0} from _llk_math_eltwise_unary_sfpu_init_once_; slot 6 is {dest.incr = 2} from
// heaviside_init above, and nothing may rewrite it between that init and this call. One caller that
// does: on WH with MM_THROTTLE set (throttle_level / TT_MM_THROTTLE_PERF, more than 48 cores),
// matmul_init / matmul_block_init program ADDR_MOD_6 with dest.clr, and minimal_matmul and
// conv_bmm_tilize run SFPU_OP_INIT_ACTIVATION only once before those inits. A heaviside fused there
// would clear the dest counter on every row and keep rewriting the tile's first row. typecast,
// square, unary max/min, signbit and the comp ops share that exposure on main; no in-tree caller
// fuses any of them there.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(std::uint32_t value) {
    // heaviside(v) = `value` at v == 0 (either zero), 0.0 for a set sign bit, 1.0 otherwise.
    //
    // Raw SFPU body, 7 instructions per row. The sfpi form (copysgn + add, then an
    // sfpi::abs(v) == 0 arm) costs 11: SFPABS + SFPSETCC for the zero test, two SFPMOVs to
    // merge the arms, and a RISC loop. Here the row is three stores: an unpredicated store of
    // `value` to every lane; then 1.0 to the lanes SFPLZ with NOSGN|CC_NE0 enables, those with
    // (bits & 0x7fffffff) != 0 -- the same |v| != 0 test as before, so -0.0 keeps `value`; then
    // 0.0 to those of them that SFPSETCC LT0 leaves enabled, the ones whose sign bit is set
    // (SFPSETCC only writes the flags of enabled lanes). Both forms dispatch on the sign bit
    // rather than a float compare, so NaN lands where it did: a NaN with a clear sign bit is
    // 1.0, with a set one 0.0. Device coverage of those classes, bit-exact: test_vif_equiv_sweep.py's
    // test_equiv_float_nonfinite (+-inf, NaN) and test_equiv_signed_zero (-0.0, +0.0).
    _sfpu_load_imm32_(p_sfpu::LREG4, value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 /*dest_reg_addr*/);
        // every lane: value
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 /*dest_reg_addr*/);
        // enable |v| != 0
        TTI_SFPLZ(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
        //   -> 1.0
        TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 /*dest_reg_addr*/);
        //   narrow to sign bit set
        TTI_SFPSETCC(0 /*imm12_math*/, p_sfpu::LREG0, 0 /*lreg_dest*/, sfpi::SFPSETCC_MOD1_LREG_LT0);
        //   -> 0.0; steps dest
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0 /*dest_reg_addr*/);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);
    }
}

}  // namespace sfpu
}  // namespace ckernel

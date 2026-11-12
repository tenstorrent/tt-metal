// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void heaviside_init() {
    // The row's last SFPSTORE steps Dest by one SFPU pass (SFP_ROWS = 2 counter rows, the step sfpi's
    // dst_reg++ takes), so the body needs no TTINCRWC. Same ADDR_MOD_6 value as init_typecast. The
    // SFPU load/store addr_mod field is 3 bits wide on Quasar, so the body names slot 6 directly.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief heaviside(v) = `value` at v == 0 (either zero), 0.0 for a set sign bit, 1.0 otherwise.
 *
 * Raw SFPU body, 7 instructions per row. The sfpi form (copysgn + add, then an sfpi::abs(v) == 0 arm)
 * cost 11 on Quasar: an SFPLOADI, SFPMOV + SFPSETSGN + SFPADDI for the sign arm, SFPABS + SFPSETCC for
 * the zero test, an SFPMOV merge, a TTINCRWC and a rolled RISC loop. Here the row is three stores: an
 * unpredicated store of `value` to every lane; then 1.0 to the lanes SFPLZ with NOSGN|CC_NE0 enables,
 * those with (bits & 0x7fffffff) != 0 -- the same |v| != 0 test as before, so -0.0 keeps `value`;
 * then 0.0 to those of them that SFPSETCC LT0 leaves enabled, the ones whose sign bit is set
 * (SFPSETCC only writes the flags of enabled lanes; imm12 bit 11 clear reads src_c as INT32, whose
 * sign is bit 31). Both forms dispatch on the sign bit rather than a float compare, so NaN lands where
 * it did: a NaN with a clear sign bit is 1.0, with a set one 0.0.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of SFPU passes (two Dest rows each) to process.
 * @param value: The value returned at v == 0, as fp32 bits.
 * @note Call @ref heaviside_init first: ADDR_MOD_7 is {dest.incr = 0} from the common SFPU init and
 *       ADDR_MOD_6 is {dest.incr = SFP_ROWS} from heaviside_init; nothing may rewrite it in between.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(const std::uint32_t value) {
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
        // every lane: value
        TTI_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
        // enable |v| != 0
        TTI_SFPLZ(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
        //   -> 1.0
        TTI_SFPSTORE(p_sfpu::LCONST_1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
        //   narrow to sign bit set
        TTI_SFPSETCC(0 /*imm12_math: src_c as INT32*/, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        //   -> 0.0; steps Dest
        TTI_SFPSTORE(p_sfpu::LCONST_0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0 /*dest_reg_addr*/);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, sfpi::SFPENCC_MOD1_EI_RI);
    }
}

}  // namespace sfpu
}  // namespace ckernel

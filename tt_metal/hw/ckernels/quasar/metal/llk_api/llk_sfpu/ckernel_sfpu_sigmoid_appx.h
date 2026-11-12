// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Quasar has no LReg-operand form of sfpi::lut: SFPLUT reads its three FP8 slope/intercept pairs from
// the LUT config registers (CREG 9-11) and its input from LReg3, so the table is loaded by
// sigmoid_appx_init. The coefficients are the Blackhole ones. The table overwrites the SFPU constants
// 0.0/1.0/-1.0 in LReg9-11, so no sfpi code that needs them may run between the init and the loop;
// the next op's init restores them.
//
// result = lut(x) + 0.5. Raw TTI rather than sfpi::lut, as on Blackhole (#65331): sfpi serialised
// every pass as load / lut / addi / store plus a TTINCRWC (5 slots per pass of SFP_ROWS rows). The
// hardware leaves the LUT destination free, so the passes alternate their result between LReg4 and
// LReg5 and pipeline: each pass's load and store sit between the previous pass's SFPLUT and its
// SFPADDI (4 slots per pass). The two-pass steady state is recorded once (executing as it records) and replayed; the
// stores walk Dest through ADDR_MOD_6 (+SFP_ROWS, programmed by sigmoid_appx_init the way init_typecast does), so no
// TTINCRWC per pass. Same SFPLUT mode and SFPADDI immediate as before.
template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    static_assert(ITERATIONS >= 4 && ITERATIONS % 2 == 0, "calculate_sigmoid_appx pipelines passes in pairs");
    constexpr unsigned HALF_BF16 = 0x3F00;  // SFPADDI immediate: 0.5f
    constexpr unsigned LUT_MODE = sfpi::SFPLUT_MOD0_SGN_RETAIN;
    constexpr unsigned NEXT_PASS = ckernel::math::SFP_ROWS;  // Dest offset of one pass

    // Passes 0 and 1 in flight.
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*pass 0*/);
    TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE);
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, NEXT_PASS /*pass 1*/);
    TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0 /*instr_mod1*/);
    // Body for passes r, r+1 (r even): load r+2, store r, lut r+2, addi r+1; then the same one pass
    // on. The two passes still unstored put each load two passes ahead of the counter.
    load_replay_buf<0 /*start*/, 8 /*len*/, true /*exec_while_loading*/>([] {
        TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 2 * NEXT_PASS /*pass r+2*/);
        TTI_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
        TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE);
        TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0 /*instr_mod1*/);
        TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 2 * NEXT_PASS /*pass r+3*/);
        TTI_SFPSTORE(p_sfpu::LREG5, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
        TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE);
        TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0 /*instr_mod1*/);
    });
#pragma GCC unroll 8
    for (int d = 4; d < ITERATIONS; d += 2) {
        TTI_REPLAY(0 /*start*/, 8 /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);
    }
    // Drain passes ITERATIONS-2 and ITERATIONS-1.
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0 /*instr_mod1*/);
    TTI_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
}

inline void sigmoid_appx_init() {
    // calculate_sigmoid_appx stores through ADDR_MOD_6: Dest += SFP_ROWS per pass, the programming
    // init_typecast uses.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);

    // Load the 3 fp8 LUT coefficient pairs into the LUT config registers. Written through
    // _sfpu_load_config32_ because the assembler rejects an sfpi SFPCONFIG to CREG 9 and 10
    // (tt-metal #51346).
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 0, 0, sfpi::sLut8si(0.22656f, 0.0f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 1, 0, sfpi::sLut8si(0.26562f, -0.04687f).get());
    math::_sfpu_load_config32_(sfpi::CREG_IDX_LUT_SLOPES + 2, 0, sfpi::sLut8si(0.0f, 0.5f).get());
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Quasar has no LReg-operand form of sfpi::lut: SFPLUT reads its three FP8 slope/intercept pairs from
// the LUT config registers (CREG 9-11) and its input from LReg3, so the table is loaded by
// sigmoid_appx_init. The coefficients are the Blackhole ones. The table overwrites the SFPU constants
// 0.0/1.0/-1.0 in LReg9-11, so no sfpi code that needs them may run between the init and the loop;
// the next op's init restores them.
//
// result = lut(x) + 0.5. A plain per-pass sfpi loop serialises every pass as load / lut / addi /
// store plus a TTINCRWC (5 slots per pass of SFP_ROWS rows). The LUT destination is free, so the
// passes alternate their result between two LRegs and pipeline: each pass's load and store sit
// between the previous pass's SFPLUT and its SFPADDI (4 slots per pass). The two-pass steady state
// is recorded once and replayed; the stores walk Dest through ADDR_MOD_6 (+SFP_ROWS, programmed by
// sigmoid_appx_init the way init_typecast does), so no TTINCRWC per pass. Same SFPLUT mode and
// SFPADDI immediate as before.
//
// The recorded body runs once per replay, so every value it carries from one pass to the next has to
// sit in the same LReg at its end as at its start. sfpi does not know the block repeats, so those
// values are pinned through l_reg[]; that costs no instruction as long as the generated body matches
// the *_REPLAY_LEN below (check the disassembly if the body or the sfpi version changes). Quasar's
// REPLAY cannot execute while it records (sfpi rejects lltt::Exec), so the body is recorded without
// executing and replayed for every pass.
inline constexpr sfpi::LRegs SIGMOID_APPX_EVEN_LREG = sfpi::LRegs::LReg4;
inline constexpr sfpi::LRegs SIGMOID_APPX_ODD_LREG = sfpi::LRegs::LReg5;
inline constexpr unsigned SIGMOID_APPX_REPLAY_START = 0;
inline constexpr unsigned SIGMOID_APPX_REPLAY_LEN = 8;

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    static_assert(ITERATIONS >= 2 && ITERATIONS % 2 == 0, "calculate_sigmoid_appx pipelines passes in pairs");

    // Passes 0 and 1 in flight.
    {
        sfpi::vFloat even = sfpi::lut<sfpi::LutMode::Fp8x3>(sfpi::dst_reg[0]);
        sfpi::l_reg[SIGMOID_APPX_ODD_LREG] = sfpi::lut<sfpi::LutMode::Fp8x3>(sfpi::dst_reg[1]);
        sfpi::l_reg[SIGMOID_APPX_EVEN_LREG] = even + 0.5f;
    }
    // Body for passes r, r+1 (r even): load r+2, store r, lut r+2, addi r+1; then the same one pass
    // on. The two passes still unstored put each load two passes ahead of the counter.
    lltt::record(SIGMOID_APPX_REPLAY_START, SIGMOID_APPX_REPLAY_LEN);
    {
        sfpi::vFloat even = sfpi::l_reg[SIGMOID_APPX_EVEN_LREG];
        sfpi::vFloat odd = sfpi::l_reg[SIGMOID_APPX_ODD_LREG];
        sfpi::vFloat next = sfpi::dst_reg[2];
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = even;
        even = sfpi::lut<sfpi::LutMode::Fp8x3>(next);
        odd += 0.5f;
        next = sfpi::dst_reg[2];
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = odd;
        sfpi::l_reg[SIGMOID_APPX_ODD_LREG] = sfpi::lut<sfpi::LutMode::Fp8x3>(next);
        sfpi::l_reg[SIGMOID_APPX_EVEN_LREG] = even + 0.5f;
    }
#pragma GCC unroll 8
    for (int d = 2; d < ITERATIONS; d += 2) {
        lltt::replay(SIGMOID_APPX_REPLAY_START, SIGMOID_APPX_REPLAY_LEN);
    }
    // Drain passes ITERATIONS-2 and ITERATIONS-1.
    {
        sfpi::vFloat even = sfpi::l_reg[SIGMOID_APPX_EVEN_LREG];
        sfpi::vFloat odd = sfpi::l_reg[SIGMOID_APPX_ODD_LREG];
        odd += 0.5f;
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = even;
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = odd;
    }
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

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// result = lut(x) + 0.5 over the 3-entry 8-bit SFPLUT table in LReg0/1/2 (sigmoid_appx_init).
//
// Raw TTI rather than sfpi::lut: sfpi writes the LUT result back into LReg3, the register SFPLUT
// reads its input from, so every row is a serial load / lut / nop / addi / nop / store chain
// (7 slots). The hardware leaves the LUT destination free, so the rows alternate their result
// between LReg4 and LReg5 and pipeline: each row's load and store fill the previous row's SFPLUT
// and SFPADDI shadows (4 slots per row, no SFPNOP). The two-row steady state is recorded once and
// replayed; the stores walk dest through ADDR_MOD_2 -- physical slot 6 under the SFPU addr-mod
// base, programmed by sigmoid_appx_init as dest += 2 -- so no TTI_INCRWC per row. Same SFPLUT
// mode and SFPADDI immediate as before; only the destination registers and the schedule change.
template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    static_assert(ITERATIONS >= 4 && ITERATIONS % 2 == 0, "calculate_sigmoid_appx pipelines rows in pairs");
    constexpr unsigned HALF_BF16 = 0x3F00;  // SFPADDI immediate: 0.5f
    constexpr unsigned LUT_MODE = sfpi::SFPLUT_MOD0_SGN_RETAIN;

    // Rows 0 and 1 in flight.
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
    TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0);
    // Body for rows r, r+1 (r even): load r+2, store r, lut r+2, addi r+1; then the same one row
    // on. The two rows still unstored put each load four dest steps ahead of the counter.
    TTI_REPLAY(0, 8, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0);
#pragma GCC unroll 8
    for (int d = 4; d < ITERATIONS; d += 2) {
        TTI_REPLAY(0, 8, 0, 0);
    }
    // Drain rows ITERATIONS-2 and ITERATIONS-1.
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
}

inline void sigmoid_appx_init() {
    // calculate_sigmoid_appx stores through ADDR_MOD_2 (physical slot 6 under the SFPU addr-mod
    // base): dest += 2 per row, the same programming typecast_init uses.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);

    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 0.5f);
}

}  // namespace sfpu
}  // namespace ckernel

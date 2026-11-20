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

// Whether the LUT approx kernels (calculate_sigmoid_appx, calculate_tanh<true>, calculate_gelu_appx)
// advance dest from their SFPSTORE through ADDR_MOD_6 (programmed by their inits) or with a
// TTI_INCRWC after it. A throttled matmul (MM_THROTTLE > 0) programs ADDR_MOD_6 for its own MOP,
// and the fused-activation matmul / conv kernels run the activation's init once, before
// matmul_init, so in a kernel built with MM_THROTTLE neither the inits nor the kernels touch it.
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
constexpr bool LUT_APPX_STORE_INCR = false;
#else
constexpr bool LUT_APPX_STORE_INCR = true;
#endif

// Issue slots of lut_appx_store_row.
constexpr std::uint32_t LUT_APPX_STORE_SLOTS = LUT_APPX_STORE_INCR ? 1 : 2;

// Stores one result row from LREG and steps dest to the next row.
template <std::uint32_t LREG>
inline void lut_appx_store_row() {
    if constexpr (LUT_APPX_STORE_INCR) {
        TTI_SFPSTORE(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    } else {
        TTI_SFPSTORE(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

// The store-side dest step of lut_appx_store_row: ADDR_MOD_2 under the SFPU addr-mod base is
// physical slot 6, dest += 2 per row, the same programming typecast_init uses.
inline void lut_appx_init_addr_mod() {
    if constexpr (LUT_APPX_STORE_INCR) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
}

// result = lut(x) + 0.5 over the 3-entry 8-bit SFPLUT table in LReg0/1/2 (sigmoid_appx_init).
//
// Raw TTI rather than sfpi::lut: sfpi writes the LUT result back into LReg3, the register SFPLUT
// reads its input from, so every row is a serial load / lut / nop / addi / nop / store chain
// (7 slots). The hardware leaves the LUT destination free, so the rows alternate their result
// between LReg4 and LReg5 and pipeline: each row's load and store fill the previous row's SFPLUT
// and SFPADDI shadows (4 slots per row, no SFPNOP). The two-row steady state is recorded once and
// replayed; the stores walk dest through ADDR_MOD_2 -- physical slot 6 under the SFPU addr-mod
// base, programmed by sigmoid_appx_init as dest += 2 -- so no TTI_INCRWC per row (one per row
// under MM_THROTTLE, see LUT_APPX_STORE_INCR). Same SFPLUT mode and SFPADDI immediate as before;
// only the destination registers and the schedule change.
template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    static_assert(ITERATIONS >= 4 && ITERATIONS % 2 == 0, "calculate_sigmoid_appx pipelines rows in pairs");
    constexpr unsigned HALF_BF16 = 0x3F00;  // SFPADDI immediate: 0.5f
    constexpr unsigned LUT_MODE = sfpi::SFPLUT_MOD0_SGN_RETAIN;
    constexpr unsigned BODY_LEN = 6 + 2 * LUT_APPX_STORE_SLOTS;

    // Rows 0 and 1 in flight.
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
    TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0);
    // Body for rows r, r+1 (r even): load r+2, store r, lut r+2, addi r+1; then the same one row
    // on. The two rows still unstored put each load four dest steps ahead of the counter.
    TTI_REPLAY(0, BODY_LEN, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    lut_appx_store_row<p_sfpu::LREG4>();
    TTI_SFPLUT(p_sfpu::LREG4, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    lut_appx_store_row<p_sfpu::LREG5>();
    TTI_SFPLUT(p_sfpu::LREG5, LUT_MODE, 0);
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG4, 0);
#pragma GCC unroll 8
    for (int d = 4; d < ITERATIONS; d += 2) {
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
    // Drain rows ITERATIONS-2 and ITERATIONS-1.
    TTI_SFPADDI(HALF_BF16, p_sfpu::LREG5, 0);
    lut_appx_store_row<p_sfpu::LREG4>();
    lut_appx_store_row<p_sfpu::LREG5>();
}

inline void sigmoid_appx_init() {
    lut_appx_init_addr_mod();

    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 0.5f);
}

}  // namespace sfpu
}  // namespace ckernel

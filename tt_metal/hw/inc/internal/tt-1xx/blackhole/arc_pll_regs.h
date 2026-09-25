// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Blackhole PLL control wrapper, in the ARC tile's XBAR space: the addresses ARC firmware uses, and the addresses a
// NoC read of the ARC tile takes. A NoC read of this space lands its data at the destination's offset modulo 64 B, so
// the L1 destination must match the register's address modulo 64.
//
// PLL0 is AICLK: AICLK = refclk * FBDIV / (REFDIV * postdiv0), postdiv0 counted as PLL_CNTL_5.postdiv0 + 1 (twice that
// above 16) when PLL_USE_POSTDIV.pll_use_postdiv0 is set, 1 when it is clear, and the clock off when postdiv0 is 0.
// Firmware changes AICLK only through FBDIV.

#define ARC_PLL_CNTL_WRAPPER_BASE 0x80020000u
#define ARC_PLL_CNTL_WRAPPER_REFCLK_PERIOD 0x8002002Cu
#define ARC_PLL_CNTL_WRAPPER_PLL_LOCK 0x80020040u

#define ARC_PLL0_BASE 0x80020100u
#define ARC_PLL1_BASE 0x80020200u
#define ARC_PLL2_BASE 0x80020300u
#define ARC_PLL3_BASE 0x80020400u
#define ARC_PLL4_BASE 0x80020500u

#define ARC_PLL_CNTL_0 0x00u
#define ARC_PLL_CNTL_1 0x04u  // refdiv [7:0], postdiv [15:8] (internal, unused), fbdiv [31:16]
#define ARC_PLL_CNTL_2 0x08u
#define ARC_PLL_CNTL_3 0x0Cu
#define ARC_PLL_CNTL_4 0x10u
#define ARC_PLL_CNTL_5 0x14u  // postdiv0..3, 8 bits each
#define ARC_PLL_CNTL_6 0x18u
#define ARC_PLL_USE_POSTDIV 0x1Cu  // pll_use_postdiv0..7 in bits 0..7
#define ARC_PLL_REFCLK_SEL 0x20u
#define ARC_PLL_USE_FINE_DIVIDER_1 0x24u
#define ARC_PLL_USE_FINE_DIVIDER_2 0x28u
#define ARC_PLL_FINE_DUTYC_ADJUST 0x2Cu
#define ARC_PLL_CLK_COUNTER_EN 0x30u
#define ARC_PLL_CLK_COUNTER_0 0x34u  // each counts its output over ARC_PLL_CNTL_WRAPPER_REFCLK_PERIOD refclk ticks
#define ARC_PLL_CLK_COUNTER_1 0x38u
#define ARC_PLL_CLK_COUNTER_2 0x3Cu
#define ARC_PLL_CLK_COUNTER_3 0x40u
#define ARC_PLL_CLK_COUNTER_4 0x44u
#define ARC_PLL_CLK_COUNTER_5 0x48u
#define ARC_PLL_CLK_COUNTER_6 0x4Cu
#define ARC_PLL_CLK_COUNTER_7 0x50u

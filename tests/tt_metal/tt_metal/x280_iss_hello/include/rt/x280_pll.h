// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ISS stub: no 50 MHz CLINT, so do not run the 8M-iteration silicon loop.

#ifndef X280_RT_PLL_H
#define X280_RT_PLL_H

#include <stdint.h>

static uint32_t __attribute__((unused)) x280_measure_pll_khz(void) { return 1000000u; }

#endif

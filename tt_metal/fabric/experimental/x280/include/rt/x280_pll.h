// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_pll.h -- measure the X280 core clock on the device.
 *
 * x280_measure_pll_khz() reports the clock the hart actually ran at, measured
 * on the device; it takes no frequency from the host and substitutes no
 * fallback constant.
 */
#ifndef X280_RT_PLL_H
#define X280_RT_PLL_H

#include <stdint.h>

#include "rt/x280_hw.h"

/* Measure the core clock in kHz against the fixed 50 MHz CLINT mtime.
 *
 * The loop must be REGISTER-ONLY (no memory access), with mtime read only at
 * the two boundaries. Polling mtime inside the loop under-counts cycles --
 * worse after heavy DMA/NOC activity has left the fabric busy, giving
 * spuriously low readings like 800 instead of 1750. rdcycle does not advance
 * while the hart stalls; a register-only loop makes no memory access, so it
 * does not stall.
 *
 *   freq = d_rdcycle / d_mtime * 50 MHz
 *
 * ~8M iterations is ~5 ms at 1.75 GHz, i.e. ~250k mtime ticks, which is ample
 * resolution. Returns 0 if mtime did not advance (nothing sane to report).
 */
static uint32_t __attribute__((unused)) x280_measure_pll_khz(void) {
    uint64_t t0 = x280_mtime();
    uint64_t c0 = x280_rdcycle();
    uint64_t n = 8000000ULL;
    __asm__ volatile(
        "1: addi %0, %0, -1\n"
        "   bnez %0, 1b\n"
        : "+r"(n)
        :
        :);
    uint64_t c1 = x280_rdcycle();
    uint64_t t1 = x280_mtime();
    uint64_t dtick = t1 - t0;
    if (dtick == 0) {
        return 0;
    }
    return (uint32_t)(((c1 - c0) * 50000ULL) / dtick);
}

#endif /* X280_RT_PLL_H */

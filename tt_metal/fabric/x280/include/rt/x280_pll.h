// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_pll.h -- measure the X280 core clock on the device.
 *
 * This is the honest alternative to the pattern the eight DRAM benchmarks use,
 * where the host writes a PLL value into a mailbox and the firmware trusts it
 * with a hardcoded fallback:
 *
 *     uint64_t pll_mhz = *X_PLL_INPUT_ADDR;
 *     if (pll_mhz == 0 || pll_mhz > 10000) pll_mhz = 1750;   // or 1000
 *
 * A fallback constant standing in for a measurement is how a mis-set mailbox
 * turns into plausible-but-wrong bandwidth numbers. Firmwares that call
 * x280_measure_pll_khz() report what the clock actually was.
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
 * while the hart stalls (MEASUREMENT.md "Unconfirmed hardware assumptions",
 * item 5); a register-only loop makes no memory access, so it does not stall.
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

/* ---------------------------------------------------------------------------
 * Host-supplied clock, with the historical fallback.
 * ---------------------------------------------------------------------------
 * The eight DRAM benchmarks do not measure the clock; the host writes it into a
 * mailbox before releasing reset and the firmware echoes it into META[0] so the
 * host can check the round-trip.
 *
 * The fallback is a wart: if the mailbox looks wrong the firmware invents a
 * clock, and the host then trusts META[0], so a mis-set mailbox yields
 * plausible-but-wrong bandwidth numbers rather than an error. Prefer
 * x280_measure_pll_khz() in new code. Removing the fallback changes what seven
 * shipping firmwares report.
 * ------------------------------------------------------------------------- */
#define X280_PLL_FALLBACK_MHZ 1750

/* A statement expression, not a static inline function: inlining a function
 * here costs one extra branch in the epilogue of four firmwares, and this
 * helper needs to be a pure textual substitution. */
#define X280_PLL_FROM_MAILBOX_OR(mbox, fallback_mhz) \
    __extension__({                                  \
        uint64_t _x280_mhz = *(mbox);                \
        if (_x280_mhz == 0 || _x280_mhz > 10000)     \
            _x280_mhz = (fallback_mhz);              \
        _x280_mhz;                                   \
    })

/* The common case: the driver writes the mailbox, so the fallback never fires.
 *
 * The fallback is per-firmware because it is NOT always dead. A firmware driven
 * by more than one host script may have one driver that writes the mailbox and
 * another that does not, and the second silently depends on the fallback value
 * matching the clock it programmed. Spell it out at the call
 * site rather than assuming 1750. */
#define X280_PLL_FROM_MAILBOX(mbox) X280_PLL_FROM_MAILBOX_OR(mbox, X280_PLL_FALLBACK_MHZ)

#endif /* X280_RT_PLL_H */

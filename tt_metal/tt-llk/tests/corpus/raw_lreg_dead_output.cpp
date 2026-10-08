// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Assembly-only experiment: preserve a typed value across an opaque raw write
// whose result is dead. This is distinct from protecting a raw producer's value
// until a later raw consumer. No device correctness claim follows from this TU.
// SCHEME: 0 none, 1 identity pair, 2 effect, 3 discarded read.
// Define USE_MMIO for TT delivery; otherwise use TTI (.ttinsn).
#ifndef SCHEME
#error "select SCHEME=0, 1, 2, or 3"
#endif
#ifndef LREG
#define LREG 0
#endif
static_assert(SCHEME >= 0 && SCHEME <= 3, "invalid scheme");
static_assert(LREG >= 0 && LREG < 8, "invalid architectural LREG");

void raw_dead_output(volatile unsigned *issue)
{
    auto before = __builtin_rvtt_sfpload(nullptr, 1, 0, 0, 0, 0);
    // SFPLOADI LREG, FLOATB, 1.0. Raw output is never subsequently observed.
    constexpr unsigned word = 0x71003f80u | (unsigned(LREG) << 20);
#if defined(USE_MMIO)
    *issue = word;
#else
    asm volatile (".ttinsn %0" : : "n"(word));
#endif
#if SCHEME == 1
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(LREG), LREG);
#elif SCHEME == 2
    __builtin_rvtt_sfprawlreg_effect(0, 1u << LREG);
#elif SCHEME == 3
    // The read's RTL is volatile, but its builtin descriptor does not carry
    // VOL. Inspect GIMPLE and RTL dumps to establish whether this unused call
    // reaches allocation; do not assume either retention or elimination.
    (void)__builtin_rvtt_sfpreadlreg(LREG);
#endif
    __builtin_rvtt_sfpstore(nullptr, before, 2, 0, 0, 0, 0);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_hw.h -- CPU primitives for the X280 baremetal firmware.
 *
 * The one place that knows how to issue memory-ordering fences, read a cycle
 * counter, read wall-clock time, hint a spin loop, and halt a hart.
 *
 * Header-only and static inline, deliberately: these are fully inlined at -O2,
 * and a real call boundary inside a measured loop would change the
 * numbers this suite exists to report. It also means C++ firmware can use it
 * without linkage games.
 */
#ifndef X280_RT_HW_H
#define X280_RT_HW_H

#include <stdint.h>

#include "x280_address_map.h"

/* --------------------------------------------------------------------------
 * Memory ordering
 * -------------------------------------------------------------------------- */

/* Orders I/O reads and writes across this point and blocks compiler motion. */
static inline __attribute__((always_inline)) void x280_fence_io(void) { __asm__ volatile("fence io, io" ::: "memory"); }

/* Orders all device and memory reads and writes and blocks compiler motion. */
static inline __attribute__((always_inline)) void x280_fence_iorw(void) {
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

/* Orders inbound reads across this point and blocks compiler motion. */
static inline __attribute__((always_inline)) void x280_fence_ir(void) { __asm__ volatile("fence ir, ir" ::: "memory"); }

/* Orders outbound writes across this point and blocks compiler motion. */
static inline __attribute__((always_inline)) void x280_fence_ow(void) { __asm__ volatile("fence ow, ow" ::: "memory"); }

/* --------------------------------------------------------------------------
 * Cycle counter -- core clock, so it scales with the PLL
 * --------------------------------------------------------------------------
 * Counts the core cycles the hart executes, at the PLL clock. It does not
 * advance while the hart is stalled on memory, a NoC transaction or a BRESP
 * (SiFive X280 MC Manual 21G3.04.00 §3.10.2.1, "clock cycles the hart has
 * executed"), so use it only for executed-cycle cost. For elapsed time and timeouts -- and in
 * particular for cross-hart overlap -- use x280_mtime() instead.
 */
static inline uint64_t x280_rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* --------------------------------------------------------------------------
 * Wall clock -- CLINT mtime, fixed 50 MHz, independent of the PLL
 * --------------------------------------------------------------------------
 * The 50 MHz rate and its independence from the PLL are an assumption: the
 * SiFive X280 MC Manual §5.13.1 defers the rtc_toggle rate to an X280 MC User
 * Guide, and no Blackhole document gives it.
 *
 * Because it is PLL-independent it is the reference used to measure the core
 * clock (see rt/x280_pll.h) and the only honest basis for timeouts, which is
 * why bounded waits should be expressed in mtime ticks rather than loop
 * iterations.
 */
#define X280_MTIME_ADDR (X280_CLINT_BASE + 0xBFF8UL)
#define X280_MTIME_HZ 50000000ULL

static inline uint64_t x280_mtime(void) { return *(volatile uint64_t*)X280_MTIME_ADDR; }

/* --------------------------------------------------------------------------
 * Spin-loop hint
 * --------------------------------------------------------------------------
 * RISC-V Zihintpause `pause` (FENCE pred=W,succ=0,fm=0 == 0x0100000F).
 * Emitted as a raw word because the stock binutils in our toolchain does not
 * assemble the `pause` mnemonic without an explicit arch option, but the X280
 * silicon *does* implement it: measured ~30 core cycles of stall per `pause`
 * vs ~0 for a NOP on this part. Using it in a spin loop keeps the core
 * quiescent (far fewer LIM accesses / loop iterations) without depending on an
 * interrupt to wake.
 */
static inline void x280_pause(void) { __asm__ volatile(".word 0x0100000F" ::: "memory"); }

/* --------------------------------------------------------------------------
 * Halt
 * --------------------------------------------------------------------------
 * SiFive CEASE. Retires all outstanding transactions and stops the hart until
 * the next reset -- there is no way back, which is what makes it the right
 * response to an unrecoverable error and the wrong response to "this hart has
 * no work". A hart that must idle but stay reachable belongs in a poll loop
 * (see rt/x280_idle.h); WFI would strand it, because host NOC writes raise no
 * interrupt on this part.
 *
 * boot/entry.S emits the same encoding at the end of its trap handler.
 *
 * The loop is inside the helper on purpose: CEASE is emitted as a raw word, so
 * the compiler cannot know it does not return, and without the loop callers
 * have to supply their own to keep the helper noreturn.
 */
static inline __attribute__((noreturn)) void x280_cease(void) {
    for (;;) {
        __asm__ volatile(".4byte 0x30500073");
    }
}

#endif /* X280_RT_HW_H */

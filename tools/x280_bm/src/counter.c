/*
 * counter.c - X280 bare-metal proof-of-life FW (step 1).
 *
 * Runs on hart 0 only (harts 1-3 park in WFI). Publishes a monotonically
 * increasing 64-bit counter to a fixed LIM address that the host polls over
 * the NOC. This is the minimal "is the firmware alive?" test: a steadily
 * rising value, read back by the host, proves the X280 booted and is
 * executing our code.
 *
 * Counter location: HB_COUNTER_ADDR = 0x08010000, just above the 64 KiB
 * region the linker manages (see ld/x280-lim.ld), so it is free LIM SRAM.
 *
 * Cadence: ~1 ms per increment (paced with rdcycle at the boot PLL of
 * 1000 MHz => 1e6 cycles/ms). A host polling every 100 ms therefore sees
 * the value climb by roughly +100 each poll. CYCLES_PER_TICK can be
 * overridden at build time.
 */
#include <stdint.h>

#define HB_COUNTER_ADDR 0x08010000UL

#ifndef CYCLES_PER_TICK
#define CYCLES_PER_TICK 1000000UL /* ~1 ms at 1000 MHz */
#endif

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

int main(uint64_t hartid) {
    /* Single-hart step 1: only hart 0 runs the heartbeat; others idle. */
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    volatile uint64_t* line = (volatile uint64_t*)HB_COUNTER_ADDR;

    /* ECC insurance: initialize the full 64-byte cache line with eight
     * aligned u64 stores before doing partial (single-word) updates. On a
     * freshly reset chip the L3 SRAM ECC bits for an untouched line are
     * uninitialized; a full-width init (the same pattern test_lim.c uses)
     * commits valid data+ECC. */
    for (int i = 0; i < 8; i++) {
        line[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    uint64_t counter = 0;
    for (;;) {
        counter++;
        line[0] = counter;
        __asm__ volatile("fence iorw, iorw");

        uint64_t start = rdcycle();
        while ((rdcycle() - start) < CYCLES_PER_TICK) {
            __asm__ volatile("nop");
        }
    }

    return 0; /* unreachable */
}

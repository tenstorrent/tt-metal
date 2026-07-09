/*
 * dma_probe.c - X280 bare-metal DMA validation + re-trigger benchmark.
 *
 * Runs on hart 0 (others park). Reads a host-provided target (a Tensix tile's
 * physical NOC coord + an L1 source address + byte count + a LIM destination +
 * a repeat count) from a LIM mailbox, then drives the X280's Synopsys DW DMAC
 * to copy that region NOC->LIM (EXTERN src master -> L2 dst master), no core
 * loads:
 *   - The FIRST transfer goes through dma_engine_noc_to_x280(), which does the
 *     full one-time setup (channel reset, CTL/CFG init, NOC-TLB program,
 *     interrupt unmask) + the transfer. Its cycle cost is reported as `cycles`.
 *   - If repeats > 1, the remaining transfers RE-TRIGGER the same descriptor
 *     without re-doing the setup: just restore block_ts + SAR/DAR, clear
 *     interrupts, re-arm the channel, assert the software handshake, poll done.
 *     The steady-state per-re-trigger cost (avg + min) is reported separately.
 * This quantifies how much of the per-transfer cost is one-time setup vs the
 * actual move -- i.e. whether "set up once, trigger over and over" pays off.
 *
 * Re-trigger assumes a single-block transfer: nbytes must be a multiple of 32
 * (the 32-byte DMA word the first transfer programs) and <= 4095*32. 2 KB fits.
 *
 * LIM mailbox layout (must match the host example):
 *   PARAMS @ 0x08011000 (host writes a full 64 B block before release):
 *     +0x00 u32 target_noc_x   +0x04 u32 target_noc_y
 *     +0x08 u64 src_l1_addr    +0x10 u64 dst_lim_addr
 *     +0x18 u64 nbytes         +0x20 u64 repeats
 *   RESULTS @ 0x08011040 (this FW zero-primes then writes):
 *     +0x00 u64 rc             +0x08 u64 cycles (first, full setup)
 *     +0x10 u32 first          +0x14 u32 last
 *     +0x18 u64 done (= DONE_MAGIC, written LAST; host polls this)
 *     +0x20 u64 retrig_avg_cycles   +0x28 u64 retrig_min_cycles
 */
#include <stdint.h>

#include "dma_engine.h" /* pulls in noc.h */

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL

#define P_TARGET_X (MBOX_PARAMS + 0x00) /* u32 */
#define P_TARGET_Y (MBOX_PARAMS + 0x04) /* u32 */
#define P_SRC_L1 (MBOX_PARAMS + 0x08)   /* u64 */
#define P_DST_LIM (MBOX_PARAMS + 0x10)  /* u64 */
#define P_NBYTES (MBOX_PARAMS + 0x18)   /* u64 */
#define P_REPEATS (MBOX_PARAMS + 0x20)  /* u64 */

#define R_RC (MBOX_RESULTS + 0x00)         /* u64 */
#define R_CYCLES (MBOX_RESULTS + 0x08)     /* u64 (first transfer) */
#define R_FIRST (MBOX_RESULTS + 0x10)      /* u32 */
#define R_LAST (MBOX_RESULTS + 0x14)       /* u32 */
#define R_DONE (MBOX_RESULTS + 0x18)       /* u64 */
#define R_RETRIG_AVG (MBOX_RESULTS + 0x20) /* u64 */
#define R_RETRIG_MIN (MBOX_RESULTS + 0x28) /* u64 */

#define DONE_MAGIC 0xDDA9C0FFEEULL
#define DEFAULT_NBYTES 2048ULL

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Lightweight re-trigger of a single-block transfer whose CTL/CFG/word-size,
 * masters, NOC TLB and interrupt-mask were all set by the first
 * dma_engine_noc_to_x280(). We only restore the per-transfer state the DMAC
 * consumes/auto-increments: block_ts and SAR/DAR. No channel reset, no CTL/CFG
 * re-init, no TLB reprogram. */
static int dma_retrigger_block(uint64_t src_dma_sar, uint64_t dst, uint32_t words32) {
    x280_dma_set_block_ts(words32);
    x280_dma_set_sar(src_dma_sar);
    x280_dma_set_dar(dst);
    x280_dma_clear_interrupts();
    x280_dma_fence_();
    x280_dma_channel_enable();
    x280_dma_fence_();
    x280_dma_start_burst();
    x280_dma_fence_();
    return x280_dma_wait_done(X280_DMA_TIMEOUT_CYCLES);
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint32_t tx = *(volatile uint32_t*)P_TARGET_X;
    uint32_t ty = *(volatile uint32_t*)P_TARGET_Y;
    uint64_t src = *(volatile uint64_t*)P_SRC_L1;
    uint64_t dst = *(volatile uint64_t*)P_DST_LIM;
    uint64_t nb = *(volatile uint64_t*)P_NBYTES;
    uint64_t repeats = *(volatile uint64_t*)P_REPEATS;
    if (nb == 0) {
        nb = DEFAULT_NBYTES;
    }
    if (repeats == 0) {
        repeats = 1;
    }

    /* Prime the results line (ECC). */
    volatile uint64_t* rline = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rline[i] = 0;
    }

    /* Zero the LIM destination first (ECC + makes a bad DMA distinguishable). */
    volatile uint64_t* d64 = (volatile uint64_t*)dst;
    for (uint64_t i = 0; i < (nb + 7) / 8; i++) {
        d64[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    /* First transfer: full one-time setup + move. */
    dma_engine_init();
    uint64_t t0 = rdcycle();
    int rc = dma_engine_noc_to_x280(tx, ty, src, X280_DMA_MASTER_L2, dst, (uint32_t)nb);
    uint64_t t1 = rdcycle();
    uint64_t cyc_first = t1 - t0;

    /* Re-triggers (same descriptor, no re-setup). Valid only for a single-block
     * 32B-aligned transfer -- the case the first call programmed. */
    uint64_t retrig_avg = 0, retrig_min = 0;
    if (rc == 0 && repeats > 1 && (nb % 32) == 0) {
        uint64_t src_dma_sar = dma_engine_noc_sar_dar(src);
        uint32_t words32 = (uint32_t)(nb / 32);
        uint64_t sum = 0, mn = ~0ULL;
        for (uint64_t k = 1; k < repeats && rc == 0; k++) {
            uint64_t a = rdcycle();
            rc = dma_retrigger_block(src_dma_sar, dst, words32);
            uint64_t b = rdcycle();
            uint64_t c = b - a;
            sum += c;
            if (c < mn) {
                mn = c;
            }
        }
        uint64_t n = repeats - 1;
        retrig_avg = sum / n;
        retrig_min = mn;
    }

    uint32_t first = *(volatile uint32_t*)dst;
    uint32_t last = *(volatile uint32_t*)(dst + nb - 4);

    *(volatile uint64_t*)R_RC = (uint64_t)rc;
    *(volatile uint64_t*)R_CYCLES = cyc_first;
    *(volatile uint32_t*)R_FIRST = first;
    *(volatile uint32_t*)R_LAST = last;
    *(volatile uint64_t*)R_RETRIG_AVG = retrig_avg;
    *(volatile uint64_t*)R_RETRIG_MIN = retrig_min;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)R_DONE = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

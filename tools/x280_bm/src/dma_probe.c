/*
 * dma_probe.c - X280 bare-metal DMA validation (DMA arc, step 1).
 *
 * Runs on hart 0 (others park). Reads a host-provided target (a Tensix tile's
 * physical NOC coord + an L1 source address + byte count + a LIM destination)
 * from a LIM mailbox, then asks the X280's Synopsys DW DMAC to copy that
 * region NOC->LIM (EXTERN src master -> L2 dst master) with NO core loads:
 * dma_engine_noc_to_x280(). It reports the DMA return code, the cycle cost,
 * and the first/last word it reads back from the LIM destination. The host
 * separately reads the LIM destination over the NOC and byte-compares it to
 * the known pattern it pre-wrote into the Tensix L1 source -- proving the
 * DMAC actually moved the bytes.
 *
 * LIM mailbox layout (must match the host example):
 *   PARAMS line @ 0x08011000 (host writes a full 64 B block before release):
 *     +0x00 u32 target_noc_x   (Tensix tile physical NOC X)
 *     +0x04 u32 target_noc_y
 *     +0x08 u64 src_l1_addr    (source address inside the Tensix tile)
 *     +0x10 u64 dst_lim_addr   (X280 LIM staging destination)
 *     +0x18 u64 nbytes
 *   RESULTS line @ 0x08011040 (this FW zero-primes then writes):
 *     +0x00 u64 rc             (0 = DMA success, 1 = err, 2 = timeout)
 *     +0x08 u64 cycles         (rdcycle delta across the DMA)
 *     +0x10 u32 first          (dst[0] read back by this FW)
 *     +0x14 u32 last           (dst[nbytes-4] read back)
 *     +0x18 u64 done           (written LAST = DONE_MAGIC; host polls this)
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

#define R_RC (MBOX_RESULTS + 0x00)     /* u64 */
#define R_CYCLES (MBOX_RESULTS + 0x08) /* u64 */
#define R_FIRST (MBOX_RESULTS + 0x10)  /* u32 */
#define R_LAST (MBOX_RESULTS + 0x14)   /* u32 */
#define R_DONE (MBOX_RESULTS + 0x18)   /* u64 */

#define DONE_MAGIC 0xDDA9C0FFEEULL
#define DEFAULT_NBYTES 256ULL

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
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
    if (nb == 0) {
        nb = DEFAULT_NBYTES;
    }

    /* Prime the results line with a full-width init (ECC). */
    volatile uint64_t* rline = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rline[i] = 0;
    }

    /* Zero the LIM destination first: primes ECC for those lines AND makes a
     * failed/partial DMA visibly distinct from the pre-written pattern. */
    volatile uint64_t* d64 = (volatile uint64_t*)dst;
    for (uint64_t i = 0; i < (nb + 7) / 8; i++) {
        d64[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    /* One-time DMAC + outbound-TLB setup, then a single NOC->LIM pull. */
    dma_engine_init();
    uint64_t t0 = rdcycle();
    int rc = dma_engine_noc_to_x280(tx, ty, src, X280_DMA_MASTER_L2, dst, (uint32_t)nb);
    uint64_t t1 = rdcycle();

    uint32_t first = *(volatile uint32_t*)dst;
    uint32_t last = *(volatile uint32_t*)(dst + nb - 4);

    *(volatile uint64_t*)R_RC = (uint64_t)rc;
    *(volatile uint64_t*)R_CYCLES = (t1 - t0);
    *(volatile uint32_t*)R_FIRST = first;
    *(volatile uint32_t*)R_LAST = last;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)R_DONE = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

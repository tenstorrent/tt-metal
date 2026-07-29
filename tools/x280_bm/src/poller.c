/*
 * poller.c - X280 bare-metal NOC poll-rate probe (step 2).
 *
 * Runs on hart 0 (harts 1-3 park). Reads a host-provided target (a Tensix
 * tile's physical NOC coord + an L1 address holding a free-running counter)
 * from a LIM mailbox, programs a 2 MiB NOC TLB window to it, then reads that
 * L1 word as fast as it can in a tight loop. Measures the elapsed core cycles
 * (rdcycle) over a fixed read count so the host can compute the poll rate
 * (ns/read, reads/s). Also captures the first and last value read, so the host
 * can confirm the X280 saw the counter advance (i.e. it polled LIVE data, not
 * a stale cached word) -- the System Port is uncached, so every load is a
 * fresh NOC round-trip.
 *
 * LIM mailbox layout (must match the host example):
 *   PARAMS line @ 0x08011000 (host writes a full 64 B block before release):
 *     +0x00 u32 target_noc_x
 *     +0x04 u32 target_noc_y
 *     +0x08 u64 l1_addr        (counter address inside the target tile)
 *     +0x10 u64 num_reads      (0 => default)
 *   RESULTS line @ 0x08011040 (this FW zero-primes then writes):
 *     +0x00 u64 reads
 *     +0x08 u64 cycles         (rdcycle delta over the read loop)
 *     +0x10 u32 first
 *     +0x14 u32 last
 *     +0x18 u64 done           (written LAST = DONE_MAGIC; host polls this)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL

#define P_TARGET_X (MBOX_PARAMS + 0x00)  /* u32 */
#define P_TARGET_Y (MBOX_PARAMS + 0x04)  /* u32 */
#define P_L1_ADDR (MBOX_PARAMS + 0x08)   /* u64 */
#define P_NUM_READS (MBOX_PARAMS + 0x10) /* u64 */

#define R_READS (MBOX_RESULTS + 0x00)  /* u64 */
#define R_CYCLES (MBOX_RESULTS + 0x08) /* u64 */
#define R_FIRST (MBOX_RESULTS + 0x10)  /* u32 */
#define R_LAST (MBOX_RESULTS + 0x14)   /* u32 */
#define R_DONE (MBOX_RESULTS + 0x18)   /* u64 */

#define DONE_MAGIC 0xD09EC0FFEEULL
#define DEFAULT_READS 2000000ULL

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

    /* Read the host-provided target from the params mailbox. */
    uint32_t tx = *(volatile uint32_t*)P_TARGET_X;
    uint32_t ty = *(volatile uint32_t*)P_TARGET_Y;
    uint64_t l1 = *(volatile uint64_t*)P_L1_ADDR;
    uint64_t n = *(volatile uint64_t*)P_NUM_READS;
    if (n == 0) {
        n = DEFAULT_READS;
    }

    /* ECC insurance: prime the results cache line with a full-width init
     * before doing the partial-word result stores (see counter.c). */
    volatile uint64_t* rline = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rline[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    /* Program a 2 MiB window to the target Tensix tile's L1 counter
     * (non-posted reads, no strict ordering). */
    volatile void* win = noc_configure_tlb_2m(0, tx, ty, l1, 0, 0);

    uint32_t first = noc_read_u32(win);

    /* Tight poll loop: read the counter word as fast as the in-order core can.
     * Each uncached load stalls until the NOC response returns; the volatile
     * read can't be elided, so all n reads issue. */
    uint32_t last = 0;
    uint64_t t0 = rdcycle();
    for (uint64_t i = 0; i < n; i++) {
        last = noc_read_u32(win);
    }
    uint64_t t1 = rdcycle();

    *(volatile uint64_t*)R_READS = n;
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

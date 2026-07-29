/*
 * poll4n1.c - X280 4-hart vector poll SPLIT across NOC0 and NOC1.
 *
 * Same as poll4 (each hart vector-polls a disjoint slice via vle64, pre-mapped
 * windows), but the last `noc1_harts` harts program their TLB windows with
 * noc_selector=1 so their reads route on NOC1 (the 2nd NIU) instead of NOC0.
 * The probe (noc1_probe) showed the SAME translated coord works on NOC1 -- only
 * the noc_selector bit changes; the System-Port read pointer is identical.
 *
 * Goal: see whether spreading the harts across the 2 NIUs exceeds the 530 MB/s
 * single-NIU (NOC0) ceiling, or whether the wall is downstream (shared mesh link
 * into the tile), in which case it stays pinned ~530.
 *
 * LIM layout (matches poll4 + noc1_harts):
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 bytes
 *                          +0x18 dst_base +0x20 nrounds +0x28 nharts +0x30 noc1_harts
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 load_reads +0x08 u64 dma_reads(0) +0x10 u64 done
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL

#define P_NUM_CORES (MBOX_PARAMS + 0x00)
#define P_SRC_L1 (MBOX_PARAMS + 0x08)
#define P_BYTES (MBOX_PARAMS + 0x10)
#define P_DST_BASE (MBOX_PARAMS + 0x18)
#define P_NROUNDS (MBOX_PARAMS + 0x20)
#define P_NHARTS (MBOX_PARAMS + 0x28)
#define P_NOC1_HARTS (MBOX_PARAMS + 0x30)
#define P_FLITS (MBOX_PARAMS + 0x38) /* flits per vector read (1 or up to 8) */

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_LOAD 0x00
#define RES_DMA 0x08
#define RES_DONE 0x10
#define DONE_MAGIC 0x4A0117C0DEULL /* same as poll4 */

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Program window `idx` to (x,y,addr) on the chosen NoC (0 or 1). */
static volatile void* cfg_win_sel(uint32_t idx, uint32_t x, uint32_t y, uint64_t addr, int sel) {
    noc_tlb_2m_t tlb;
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.data[3] = 0;
    tlb.addr = addr >> 21;
    tlb.x_end = x;
    tlb.y_end = y;
    tlb.x_start = x;
    tlb.y_start = y;
    tlb.noc_selector = (uint64_t)sel;
    return noc_configure_tlb_2m_ext(idx, &tlb, 0);
}

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t noc1_harts = *(volatile uint64_t*)P_NOC1_HARTS;
    uint64_t flits = *(volatile uint64_t*)P_FLITS;
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    if (noc1_harts > nharts) {
        noc1_harts = nharts;
    }
    if (flits == 0 || flits > 8) {
        flits = 1;
    }
    uint64_t avl = flits * 8; /* e64 elements; m8 -> VLMAX 64 = 8 flits */
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t slot = RES_SLOT(hartid);

    if (hartid >= nharts) {
        *(volatile uint64_t*)(slot + RES_LOAD) = 0;
        *(volatile uint64_t*)(slot + RES_DMA) = 0;
        __asm__ volatile("fence iorw, iorw");
        *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    /* Last noc1_harts harts use NOC1; the rest NOC0. */
    int my_noc = (hartid >= (nharts - noc1_harts)) ? 1 : 0;

    uint64_t q = (num_cores + nharts - 1) / nharts;
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }

    volatile uint64_t* rl = (volatile uint64_t*)slot;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    /* Pre-map my slice's windows on my NoC (index = core index). */
    for (uint64_t c = lo; c < hi; c++) {
        (void)cfg_win_sel((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], src_l1, my_noc);
    }
    __asm__ volatile("fence iorw, iorw");

    uint64_t load_reads = 0, sink = 0;
    uint64_t t0 = rdcycle();
    for (uint64_t round = 0; round < nrounds; round++) {
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t ptr = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
            uint64_t a0;
            /* vl = avl (flits*8 e64) capped at VLMAX 64 (m8) -> read `flits` 64 B
             * flits in one vector load = `flits` NoC reads in flight. */
            __asm__ volatile(
                "vsetvli zero, %2, e64, m8, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vmv.x.s %0, v0\n"
                : "=r"(a0)
                : "r"(ptr), "r"(avl)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
            sink ^= a0;
            *(volatile uint64_t*)(dst_base + c * 8) = a0;
            load_reads++;
        }
    }
    uint64_t t1 = rdcycle();

    *(volatile uint64_t*)(slot + RES_LOAD) = load_reads;
    *(volatile uint64_t*)(slot + RES_DMA) = (t1 - t0); /* reuse: per-hart cycles */
    (void)sink;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

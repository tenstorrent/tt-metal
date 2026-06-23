/*
 * poll4.c - X280 bare-metal multi-hart grid poll (NO DMA).
 *
 * ALL FOUR harts run concurrently; each busy-polls 1/nharts of the worker grid
 * over the NoC using core loads (no DMA). Per core it reads one 64 B flit as 8
 * independent u64 in one expression (the in-flight-overlap trick: 8 loads to the
 * same flit coalesce into ~one NoC transaction, vs serial stalls). Each hart owns
 * its own NoC TLB index (= hartid) + System-Port window, so the 4 harts never
 * clobber each other's TLB config. This is the bare-metal version of the Linux
 * `pollall`; bare-metal frees all 4 harts (Linux had to leave 1 for the OS), so
 * the question is whether 4 polling harts match/exceed the ~430 MB/s NoC-port
 * ceiling that 3 Linux harts hit.
 *
 * LIM layout (must match the host):
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 bytes(=64)
 *                          +0x18 dst_base  +0x20 nrounds +0x28 nharts
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 cycles  +0x08 u32 cores_done
 *                          +0x10 u64 sink    +0x18 u64 done (= DONE_MAGIC, last)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (virtual coords)
 *   DST     @ dst_base    : num_cores x 16 B { u64 w0(counter), u64 w1(tag) }
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

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_CYCLES 0x00
#define RES_CORES 0x08
#define RES_SINK 0x10
#define RES_DONE 0x18

#define DONE_MAGIC 0x4A0117C0DEULL

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;

    uint64_t slot = RES_SLOT(hartid);

    /* Harts beyond nharts (or with an empty slice) just stamp done and park. */
    if (hartid >= nharts) {
        *(volatile uint64_t*)(slot + RES_CYCLES) = 0;
        *(volatile uint32_t*)(slot + RES_CORES) = 0;
        *(volatile uint64_t*)(slot + RES_SINK) = 0;
        __asm__ volatile("fence iorw, iorw");
        *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    /* This hart's contiguous slice of the grid. */
    uint64_t q = (num_cores + nharts - 1) / nharts; /* ceil */
    uint64_t lo = hartid * q;
    uint64_t hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }

    /* Prime this hart's result slot (full-line write, ECC). */
    volatile uint64_t* rl = (volatile uint64_t*)slot;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    uint64_t sink = 0;
    uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);

    /* PRE-MAP one TLB window per core in my slice (window index = core index,
     * disjoint across harts; 110 cores <= 224 windows). Done ONCE, before timing:
     * the poll loop then only READS (reprogramming the TLB per read was the
     * throughput killer -- 2 peripheral-port fences + 4 cfg writes per core). */
    for (uint64_t c = lo; c < hi; c++) {
        (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], src_l1, 0, 0);
    }
    __asm__ volatile("fence iorw, iorw");

    /* vl = 8 x e64 = 64 B = one flit, read in a single vector load (one NoC
     * transaction). Scalar 8x u64 loads serialize on the in-order core (~8 NoC
     * round-trips); vle64.v fetches the whole flit at once. */
    uint64_t t0 = rdcycle();
    for (uint64_t round = 0; round < nrounds; round++) {
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t ptr = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
            uint64_t a0;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n" /* 64 B flit -> v0 in one transaction */
                "vmv.x.s %0, v0\n"   /* a0 = v0[0] = {counter(lo32), tag(hi32)} */
                : "=r"(a0)
                : "r"(ptr)
                : "memory", "v0");
            sink ^= a0;
            *(volatile uint64_t*)(dst_base + c * 8) = a0;
        }
    }
    uint64_t t1 = rdcycle();

    *(volatile uint64_t*)(slot + RES_CYCLES) = (t1 - t0);
    *(volatile uint32_t*)(slot + RES_CORES) = (uint32_t)(hi - lo);
    *(volatile uint64_t*)(slot + RES_SINK) = sink;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

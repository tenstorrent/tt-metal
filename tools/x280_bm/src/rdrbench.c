/*
 * rdrbench.c - X280 READER-DRAIN microbenchmark, vector-WIDTH sweep.
 *
 * Isolates the round-robin READER (no relay/D2H): N reader harts each own a disjoint slice of the core
 * grid and, per core, POLL the 16-word control region then DRAIN K markers (K*mw words) from that core's
 * L1 buffer into the hart's own streaming 256 KiB LIM SPSC ring. Single outstanding read (ILP=1, proven a
 * no-op) -- the lever under test here is the WIDTH of each NoC read: m1 = 64 B/read vs m8 = 512 B/read
 * (8-register vector group). If per-hart throughput is limited by per-transaction fixed cost, wider reads
 * amortize it; if it is ingress bandwidth, m8 == m1.
 *
 * Sweeps WIDTH = {m1, m8} x K = {0,4,8,16,32,64,128,256,512} in ONE boot. Results (per-hart, per-(wi,K)
 * cycle counts) land in a dedicated LIM region (BENCH_RES); the host computes drain rate.
 *
 * LIM layout:
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 mw +0x18 nrounds +0x20 nharts
 *                          +0x28 dst_base +0x30 noc_split(reader h -> NOC h&1)
 *   BENCH_RES @ 0x08013000 : per hart h @ +h*RES_STRIDE: NWID*NK u64 cycles then done magic.
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated).
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_COORDS 0x08011200UL
#define BENCH_RES 0x08013000UL /* dedicated results region (2D grid is bigger than the mailbox) */

#define P_NUM_CORES (MBOX_PARAMS + 0x00)
#define P_SRC_L1 (MBOX_PARAMS + 0x08)
#define P_MW (MBOX_PARAMS + 0x10)
#define P_NROUNDS (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_DST_BASE (MBOX_PARAMS + 0x28)
#define P_NOC_SPLIT (MBOX_PARAMS + 0x30)

#define DONE_MAGIC 0x5EADE5511BULL

#define NWID 2
static const uint32_t WIDTHLIST[NWID] = {1u, 8u}; /* vector LMUL: m1 = 64 B/read, m8 = 512 B/read */
#define NK 9
static const uint32_t KLIST[NK] = {0u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u};

#define RES_STRIDE 0x180u /* NWID*NK=18 u64 cycles + done, padded */
#define RES_SLOT(h) (BENCH_RES + (uint64_t)(h) * RES_STRIDE)
#define RES_CELL(h, wi, ki) (RES_SLOT(h) + (uint64_t)((wi) * NK + (ki)) * 8)
#define RES_DONE(h) (RES_SLOT(h) + (uint64_t)(NWID * NK) * 8)

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* m1: read one 64 B flit (8 e64, VLEN=512) from `p` into v0, store to `s`. Single outstanding read. */
static inline void rd1(uint64_t p, uint64_t s) {
    __asm__ volatile("vsetivli zero,8,e64,m1,ta,ma\n vle64.v v0,(%0)\n vse64.v v0,(%1)\n"
                     :
                     : "r"(p), "r"(s)
                     : "memory", "v0");
}

/* m8: read one 512 B chunk (64 e64 = 8 contiguous 64 B flits, VLEN=512) from `p` into the v0-v7 group,
 * store to `s`. One wide NoC transaction instead of eight -- the width lever under test. */
static inline void rdm8(uint64_t p, uint64_t s) {
    uint64_t vl = 64;
    __asm__ volatile("vsetvli zero,%2,e64,m8,ta,ma\n vle64.v v0,(%0)\n vse64.v v0,(%1)\n"
                     :
                     : "r"(p), "r"(s), "r"(vl)
                     : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t mw = *(volatile uint64_t*)P_MW;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t noc_split = *(volatile uint64_t*)P_NOC_SPLIT;
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (nharts == 0 || nharts > 4) {
        nharts = 2;
    }
    if (mw == 0) {
        mw = 4;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);

    if (hartid >= nharts) {
        __asm__ volatile("fence iorw, iorw");
        *(volatile uint64_t*)RES_DONE(hartid) = DONE_MAGIC;
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint64_t q = (num_cores + nharts - 1) / nharts;
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }

    /* Each reader writes its drained markers into its OWN 256 KiB SPSC ring (matches the real reader's
     * RSPSC_NPAGE*64), advancing a producer offset so writes stream to FRESH LIM addresses (wrapping) --
     * modelling the real SPSC-write cost, not a hot overwritten scratch. The ctrl POLL is discarded to a
     * separate small scratch (the real reader never writes ctrl into the SPSC). */
    const uint64_t RB_SPSC_SIZE = 0x40000ull;        /* 256 KiB per-reader SPSC ring */
    const uint64_t RB_USABLE = RB_SPSC_SIZE - 0x400; /* cap the modulo so an ILP group never straddles */
    uint64_t spsc = dst_base + hartid * 0x42000ull;  /* per-hart region = 256 KiB SPSC + 8 KiB scratch */
    uint64_t pscr = spsc + RB_SPSC_SIZE;             /* poll-discard scratch */

    /* Pre-map one 2 MiB window per core; reader h reads over NOC (h&1) if noc_split. */
    uint32_t my_noc = noc_split ? (uint32_t)(hartid & 1u) : 0u;
    for (uint64_t c = lo; c < hi; c++) {
        noc_tlb_2m_t wt;
        wt.data[0] = 0;
        wt.data[1] = 0;
        wt.data[2] = 0;
        wt.data[3] = 0;
        wt.addr = src_l1 >> 21;
        wt.x_end = coords[c * 2 + 0];
        wt.y_end = coords[c * 2 + 1];
        wt.x_start = coords[c * 2 + 0];
        wt.y_start = coords[c * 2 + 1];
        wt.noc_selector = my_noc;
        (void)noc_configure_tlb_2m_ext((uint32_t)c, &wt, 0);
    }
    __asm__ volatile("fence iorw, iorw");

    for (int wi = 0; wi < NWID; wi++) {
        uint32_t lmul = WIDTHLIST[wi]; /* 1 = m1 (64 B/read), 8 = m8 (512 B/read) */
        for (int ki = 0; ki < NK; ki++) {
            uint32_t K = KLIST[ki];
            uint32_t mflits = (uint32_t)(K * mw) / 16u; /* 16 words = 64 B = one flit */
            uint64_t wprod = 0;                         /* streaming SPSC producer offset (bytes) */
            uint64_t t0 = rdcycle();
            for (uint64_t round = 0; round < nrounds; round++) {
                for (uint64_t c = lo; c < hi; c++) {
                    uint64_t wb = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                    rd1(wb, pscr); /* POLL: 64 B ctrl flit -> discard scratch */
                    uint64_t base = wb + 128;
                    uint32_t f = 0;
                    if (lmul == 8) {
                        /* DRAIN wide: 512 B (8 flits) per NoC read, streaming to fresh SPSC slots. */
                        for (; f + 8 <= mflits; f += 8) {
                            rdm8(base + (uint64_t)f * 64, spsc + (wprod % RB_USABLE));
                            wprod += 512;
                        }
                    }
                    /* DRAIN narrow: 64 B/read (all of m1; the <8-flit tail of m8). */
                    for (; f < mflits; f++) {
                        rd1(base + (uint64_t)f * 64, spsc + (wprod % RB_USABLE));
                        wprod += 64;
                    }
                }
            }
            uint64_t t1 = rdcycle();
            *(volatile uint64_t*)RES_CELL(hartid, wi, ki) = (t1 - t0);
        }
    }
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)RES_DONE(hartid) = DONE_MAGIC;
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

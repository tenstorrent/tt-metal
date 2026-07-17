/*
 * rdrbench.c - X280 READER-DRAIN microbenchmark with ILP sweep.
 *
 * Isolates the round-robin READER (no relay/SPSC/D2H): N reader harts each own a disjoint slice of the
 * core grid and, per core, POLL the 16-word control region then DRAIN K markers (K*mw words) from that
 * core's single L1 buffer into a per-hart LIM sink. The reads are issued with instruction-level
 * PARALLELISM: `ilp` cores' 64 B flits are launched in flight (ilp independent NoC reads) before draining
 * them, overlapping the per-transaction NoC latency (the lever from x280_baremetal_fw_step1).
 *
 * Sweeps K = {0,4,8,16,...,4096,5000} in ONE boot at ILP=1 (single outstanding read). Results (per-hart,
 * per-(ilp,K) cycle counts) land in a dedicated LIM region (BENCH_RES); the host computes drain rate.
 *
 * LIM layout:
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 mw +0x18 nrounds +0x20 nharts
 *                          +0x28 dst_base +0x30 noc_split(reader h -> NOC h&1)
 *   BENCH_RES @ 0x08013000 : per hart h @ +h*RES_STRIDE: NILP*NK u64 cycles then done magic.
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

#define NILP 1
static const uint32_t ILPLIST[NILP] = {1u};
#define NK 13
static const uint32_t KLIST[NK] = {0u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 5000u};

#define RES_STRIDE 0x180u /* NILP*NK=13 u64 cycles + done, padded */
#define RES_SLOT(h) (BENCH_RES + (uint64_t)(h) * RES_STRIDE)
#define RES_CELL(h, ii, ki) (RES_SLOT(h) + (uint64_t)((ii) * NK + (ki)) * 8)
#define RES_DONE(h) (RES_SLOT(h) + (uint64_t)(NILP * NK) * 8)

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Read `ilp` 64 B flits (8 e64 each) from `ilp` DIFFERENT windows in flight, then store them -- the vle's
 * are all issued before the vse's, so `ilp` NoC reads are outstanding at once. sink[j] = ptr[j]. */
static inline void rd1(uint64_t p0, uint64_t s) {
    __asm__ volatile("vsetivli zero,8,e64,m1,ta,ma\n vle64.v v0,(%0)\n vse64.v v0,(%1)\n"
                     :
                     : "r"(p0), "r"(s)
                     : "memory", "v0");
}
static inline void rd2(uint64_t p0, uint64_t p1, uint64_t s) {
    __asm__ volatile(
        "vsetivli zero,8,e64,m1,ta,ma\n vle64.v v0,(%0)\n vle64.v v1,(%1)\n"
        "vse64.v v0,(%2)\n vse64.v v1,(%3)\n"
        :
        : "r"(p0), "r"(p1), "r"(s), "r"(s + 64)
        : "memory", "v0", "v1");
}
static inline void rd4(uint64_t p0, uint64_t p1, uint64_t p2, uint64_t p3, uint64_t s) {
    __asm__ volatile(
        "vsetivli zero,8,e64,m1,ta,ma\n"
        "vle64.v v0,(%0)\n vle64.v v1,(%1)\n vle64.v v2,(%2)\n vle64.v v3,(%3)\n"
        "vse64.v v0,(%4)\n vse64.v v1,(%5)\n vse64.v v2,(%6)\n vse64.v v3,(%7)\n"
        :
        : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(s), "r"(s + 64), "r"(s + 128), "r"(s + 192)
        : "memory", "v0", "v1", "v2", "v3");
}
static inline void rd8(uint64_t b, uint64_t stride, uint64_t s) {
    /* 8 windows at b, b+stride, ... b+7*stride; sink at s, s+64, ... */
    __asm__ volatile(
        "vsetivli zero,8,e64,m1,ta,ma\n"
        "vle64.v v0,(%0)\n vle64.v v1,(%1)\n vle64.v v2,(%2)\n vle64.v v3,(%3)\n"
        "vle64.v v4,(%4)\n vle64.v v5,(%5)\n vle64.v v6,(%6)\n vle64.v v7,(%7)\n"
        "vse64.v v0,(%8)\n vse64.v v1,(%9)\n vse64.v v2,(%10)\n vse64.v v3,(%11)\n"
        "vse64.v v4,(%12)\n vse64.v v5,(%13)\n vse64.v v6,(%14)\n vse64.v v7,(%15)\n"
        :
        : "r"(b),
          "r"(b + stride),
          "r"(b + 2 * stride),
          "r"(b + 3 * stride),
          "r"(b + 4 * stride),
          "r"(b + 5 * stride),
          "r"(b + 6 * stride),
          "r"(b + 7 * stride),
          "r"(s),
          "r"(s + 64),
          "r"(s + 128),
          "r"(s + 192),
          "r"(s + 256),
          "r"(s + 320),
          "r"(s + 384),
          "r"(s + 448)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}
static inline void rd16(uint64_t b, uint64_t stride, uint64_t s) {
    rd8(b, stride, s);
    rd8(b + 8 * stride, stride, s + 512);
}

/* Read one 64 B flit from each of `ilp` consecutive windows (base `wb`, step `wstride`) at byte offset
 * `roff` into that window, in flight. Handles the general ilp; falls back to rd1 for a short tail. */
static inline void read_group(uint32_t ilp, uint64_t wb, uint64_t wstride, uint64_t roff, uint64_t sink) {
    uint64_t p = wb + roff;
    if (ilp == 1) {
        rd1(p, sink);
    } else if (ilp == 2) {
        rd2(p, p + wstride, sink);
    } else if (ilp == 4) {
        rd4(p, p + wstride, p + 2 * wstride, p + 3 * wstride, sink);
    } else if (ilp == 8) {
        rd8(p, wstride, sink);
    } else { /* 16 */
        rd16(p, wstride, sink);
    }
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

    for (int ii = 0; ii < NILP; ii++) {
        uint32_t ilp = ILPLIST[ii];
        for (int ki = 0; ki < NK; ki++) {
            uint32_t K = KLIST[ki];
            uint32_t mflits = (uint32_t)(K * mw) / 16u; /* 16 words = 64 B = one flit */
            uint64_t wprod = 0;                         /* streaming SPSC producer offset (bytes) */
            uint64_t t0 = rdcycle();
            for (uint64_t round = 0; round < nrounds; round++) {
                uint64_t c = lo;
                /* Full ILP groups. Poll -> discard scratch; drain -> streaming SPSC (fresh slots). */
                for (; c + ilp <= hi; c += ilp) {
                    uint64_t wb = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                    read_group(ilp, wb, NOC_2M_WINDOW_STRIDE, 0, pscr); /* POLL: ctrl flit -> discard */
                    for (uint32_t f = 0; f < mflits; f++) {             /* DRAIN: marker flits -> SPSC */
                        read_group(ilp, wb, NOC_2M_WINDOW_STRIDE, 128 + (uint64_t)f * 64, spsc + (wprod % RB_USABLE));
                        wprod += (uint64_t)ilp * 64;
                    }
                }
                /* Tail (< ilp cores left): 1 outstanding each. */
                for (; c < hi; c++) {
                    uint64_t p = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                    rd1(p, pscr);
                    for (uint32_t f = 0; f < mflits; f++) {
                        rd1(p + 128 + (uint64_t)f * 64, spsc + (wprod % RB_USABLE));
                        wprod += 64;
                    }
                }
            }
            uint64_t t1 = rdcycle();
            *(volatile uint64_t*)RES_CELL(hartid, ii, ki) = (t1 - t0);
        }
    }
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)RES_DONE(hartid) = DONE_MAGIC;
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

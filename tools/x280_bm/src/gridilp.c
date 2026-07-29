/*
 * gridilp.c - X280 full-grid scatter poll with INSTRUCTION-LEVEL PARALLELISM.
 *
 * This is poll4 (4 harts each drain 1/nharts of the 110-core grid, 1 flit/core
 * over the NoC) PLUS the lever found in §11 (pollmp): instead of read one core ->
 * consume -> store -> next (one outstanding NoC read per hart, the 135 MB/s /
 * 530 MB/s "wall"), each hart issues `ilp` INDEPENDENT 64 B reads to `ilp`
 * DIFFERENT cores' windows, then drains them. The question: does the scatter
 * profiler pattern overlap reads across windows the way the sequential stream did
 * (793 MB/s/hart), or is cross-window scatter different?
 *
 * Full grid is always covered (slice division), so identity + liveness still
 * verify against the live BRISC counters/tags, exactly like poll4.
 *
 * LIM layout (matches poll4 + ilp):
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 bytes(=64)
 *                          +0x18 dst_base  +0x20 nrounds +0x28 nharts +0x30 ilp
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 cycles +0x08 u32 cores_done
 *                          +0x10 u64 sink   +0x18 u64 done (= DONE_MAGIC, last)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   DST     @ dst_base    : num_cores x 8 B { u64 = counter(lo32) | tag(hi32) }
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
#define P_ILP (MBOX_PARAMS + 0x30)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_CYCLES 0x00
#define RES_CORES 0x08
#define RES_SINK 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0x6217D1119DEULL

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Read one 64 B flit from window `c` (one NoC read), store its first u64 to dst. */
#define RD1(c)                                                               \
    do {                                                                     \
        uint64_t p0 = NOC_2M_WINDOW_BASE + (c) * NOC_2M_WINDOW_STRIDE + off; \
        uint64_t a0;                                                         \
        __asm__ volatile(                                                    \
            "vsetivli zero, 8, e64, m1, ta, ma\n"                            \
            "vle64.v v0, (%1)\n"                                             \
            "vmv.x.s %0, v0\n"                                               \
            : "=r"(a0)                                                       \
            : "r"(p0)                                                        \
            : "memory", "v0");                                               \
        sink ^= a0;                                                          \
        *(volatile uint64_t*)(dst_base + (c) * 8) = a0;                      \
    } while (0)

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t ilp = *(volatile uint64_t*)P_ILP;
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    if (ilp != 1 && ilp != 2 && ilp != 4 && ilp != 8) {
        ilp = 4;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t slot = RES_SLOT(hartid);
    uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);

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

    /* Pre-map one window per core in my slice (index = core index), once. */
    for (uint64_t c = lo; c < hi; c++) {
        (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], src_l1, 0, 0);
    }
    __asm__ volatile("fence iorw, iorw");

    uint64_t sink = 0;
    uint64_t t0 = rdcycle();
    for (uint64_t round = 0; round < nrounds; round++) {
        uint64_t c = lo;
        if (ilp == 1) {
            for (; c < hi; c++) {
                RD1(c);
            }
        } else if (ilp == 2) {
            for (; c + 2 <= hi; c += 2) {
                uint64_t p0 = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                uint64_t p1 = NOC_2M_WINDOW_BASE + (c + 1) * NOC_2M_WINDOW_STRIDE + off;
                uint64_t a0, a1;
                /* 2 independent NoC reads in flight, then drain both. */
                __asm__ volatile(
                    "vsetivli zero, 8, e64, m1, ta, ma\n"
                    "vle64.v v0, (%2)\n"
                    "vle64.v v1, (%3)\n"
                    "vmv.x.s %0, v0\n"
                    "vmv.x.s %1, v1\n"
                    : "=r"(a0), "=r"(a1)
                    : "r"(p0), "r"(p1)
                    : "memory", "v0", "v1");
                sink ^= a0 ^ a1;
                *(volatile uint64_t*)(dst_base + c * 8) = a0;
                *(volatile uint64_t*)(dst_base + (c + 1) * 8) = a1;
            }
            for (; c < hi; c++) {
                RD1(c);
            }
        } else if (ilp == 4) {
            for (; c + 4 <= hi; c += 4) {
                uint64_t p0 = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                uint64_t p1 = NOC_2M_WINDOW_BASE + (c + 1) * NOC_2M_WINDOW_STRIDE + off;
                uint64_t p2 = NOC_2M_WINDOW_BASE + (c + 2) * NOC_2M_WINDOW_STRIDE + off;
                uint64_t p3 = NOC_2M_WINDOW_BASE + (c + 3) * NOC_2M_WINDOW_STRIDE + off;
                uint64_t a0, a1, a2, a3;
                __asm__ volatile(
                    "vsetivli zero, 8, e64, m1, ta, ma\n"
                    "vle64.v v0, (%4)\n"
                    "vle64.v v1, (%5)\n"
                    "vle64.v v2, (%6)\n"
                    "vle64.v v3, (%7)\n"
                    "vmv.x.s %0, v0\n"
                    "vmv.x.s %1, v1\n"
                    "vmv.x.s %2, v2\n"
                    "vmv.x.s %3, v3\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(p0), "r"(p1), "r"(p2), "r"(p3)
                    : "memory", "v0", "v1", "v2", "v3");
                sink ^= a0 ^ a1 ^ a2 ^ a3;
                *(volatile uint64_t*)(dst_base + c * 8) = a0;
                *(volatile uint64_t*)(dst_base + (c + 1) * 8) = a1;
                *(volatile uint64_t*)(dst_base + (c + 2) * 8) = a2;
                *(volatile uint64_t*)(dst_base + (c + 3) * 8) = a3;
            }
            for (; c < hi; c++) {
                RD1(c);
            }
        } else { /* ilp == 8 */
            for (; c + 8 <= hi; c += 8) {
                uint64_t pb = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
                uint64_t s = NOC_2M_WINDOW_STRIDE;
                uint64_t p0 = pb, p1 = pb + s, p2 = pb + 2 * s, p3 = pb + 3 * s;
                uint64_t p4 = pb + 4 * s, p5 = pb + 5 * s, p6 = pb + 6 * s, p7 = pb + 7 * s;
                uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
                __asm__ volatile(
                    "vsetivli zero, 8, e64, m1, ta, ma\n"
                    "vle64.v v0, (%8)\n"
                    "vle64.v v1, (%9)\n"
                    "vle64.v v2, (%10)\n"
                    "vle64.v v3, (%11)\n"
                    "vle64.v v4, (%12)\n"
                    "vle64.v v5, (%13)\n"
                    "vle64.v v6, (%14)\n"
                    "vle64.v v7, (%15)\n"
                    "vmv.x.s %0, v0\n"
                    "vmv.x.s %1, v1\n"
                    "vmv.x.s %2, v2\n"
                    "vmv.x.s %3, v3\n"
                    "vmv.x.s %4, v4\n"
                    "vmv.x.s %5, v5\n"
                    "vmv.x.s %6, v6\n"
                    "vmv.x.s %7, v7\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7)
                    : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6), "r"(p7)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
                *(volatile uint64_t*)(dst_base + c * 8) = a0;
                *(volatile uint64_t*)(dst_base + (c + 1) * 8) = a1;
                *(volatile uint64_t*)(dst_base + (c + 2) * 8) = a2;
                *(volatile uint64_t*)(dst_base + (c + 3) * 8) = a3;
                *(volatile uint64_t*)(dst_base + (c + 4) * 8) = a4;
                *(volatile uint64_t*)(dst_base + (c + 5) * 8) = a5;
                *(volatile uint64_t*)(dst_base + (c + 6) * 8) = a6;
                *(volatile uint64_t*)(dst_base + (c + 7) * 8) = a7;
            }
            for (; c < hi; c++) {
                RD1(c);
            }
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

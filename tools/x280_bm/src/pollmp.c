/*
 * pollmp.c - X280 long-shot levers vs the 530 MB/s mesh-ingress wall.
 *
 * Two ideas the earlier sweeps left untested, both aimed at getting MORE THAN ONE
 * NoC read in flight per hart (the uncached System Port issues one at a time --
 * that is why a wide vector load serialised and why a single hart caps at
 * ~135 MB/s):
 *
 *   #3 Cached Memory Port. Read the SAME NoC TLB window through the cacheable,
 *      coherent Memory Port alias (NOC_2M_WINDOW_BASE_MEMPORT) instead of the
 *      uncached System Port. The L2 can then have several outstanding line fills
 *      (MSHRs) + a hardware prefetcher, so a single hart could pull multiple NoC
 *      reads concurrently. To measure REAL NoC traffic (not cache hits) each hart
 *      STREAMS a large distinct linear region in a single pass -- no line is ever
 *      re-read, which is also the best case for the prefetcher.
 *   #4 Static VC. Pin each hart's window to its own NoC virtual channel
 *      (static_en + static_vc) in case strict single-VC ordering is what limits
 *      concurrency.
 *
 * Inner loop reads 4 independent 64 B lines per iteration (v0..v3, distinct
 * address regs) so the LSU has 4 independent loads to overlap on the cached port.
 *
 * Each hart streams from ITS OWN core (index = hartid) through ITS OWN window
 * (index = hartid), so harts never share TLB config or NoC source.
 *
 * LIM layout (must match the host):
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 span_bytes
 *                          +0x18 dst_base  +0x20 nharts +0x28 memport(0/1)
 *                          +0x30 vc (0xFFFFFFFF=none, 0xFFFFFFFE=spread vc=hartid,
 *                                    else fixed 0..7)
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 cycles +0x08 u64 bytes
 *                          +0x10 u64 sink   +0x18 u64 done (= DONE_MAGIC, last)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL

#define P_NUM_CORES (MBOX_PARAMS + 0x00)
#define P_SRC_L1 (MBOX_PARAMS + 0x08)
#define P_SPAN (MBOX_PARAMS + 0x10)
#define P_DST_BASE (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_MEMPORT (MBOX_PARAMS + 0x28)
#define P_VC (MBOX_PARAMS + 0x30)
#define P_ILP (MBOX_PARAMS + 0x38) /* independent loads in flight: 1, 2, or 4 */

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_CYCLES 0x00
#define RES_BYTES 0x08
#define RES_SINK 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0x4D90111CDEULL

#define VC_NONE 0xFFFFFFFFu
#define VC_SPREAD 0xFFFFFFFEu

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Program window `idx` to (x,y,addr) on NOC0, choosing System vs Memory port and
 * optionally pinning a static VC. Returns the access pointer for that port. */
static volatile void* cfg_win(uint32_t idx, uint32_t x, uint32_t y, uint64_t addr, int memport, uint32_t vc) {
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
    tlb.noc_selector = 0;
    if (vc <= 7) {
        tlb.static_en = 1;
        tlb.static_vc = vc;
    }
    return noc_configure_tlb_2m_ext(idx, &tlb, memport);
}

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t span = *(volatile uint64_t*)P_SPAN;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t memport = *(volatile uint64_t*)P_MEMPORT;
    uint32_t vc_param = (uint32_t)*(volatile uint64_t*)P_VC;
    uint64_t ilp = *(volatile uint64_t*)P_ILP;
    (void)num_cores;
    if (ilp != 1 && ilp != 2 && ilp != 4 && ilp != 8) {
        ilp = 4;
    }
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    if (span < 512) {
        span = 512;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t slot = RES_SLOT(hartid);

    volatile uint64_t* rl = (volatile uint64_t*)slot;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    if (hartid >= nharts) {
        __asm__ volatile("fence iorw, iorw");
        *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint32_t myvc;
    if (vc_param == VC_NONE) {
        myvc = 0xFFu; /* don't pin */
    } else if (vc_param == VC_SPREAD) {
        myvc = (uint32_t)(hartid & 7);
    } else {
        myvc = vc_param & 7u;
    }

    uint32_t cx = coords[hartid * 2 + 0];
    uint32_t cy = coords[hartid * 2 + 1];
    volatile void* p = cfg_win((uint32_t)hartid, cx, cy, src_l1, (int)memport, myvc);
    __asm__ volatile("fence iorw, iorw");
    uint64_t base = (uint64_t)p;

    uint64_t nlines = span / 64; /* 64 B lines / flits */
    nlines &= ~7ULL;             /* multiple of 8 */
    if (nlines == 0) {
        nlines = 8;
    }
    uint64_t sink = 0;

    /* `ilp` independent line loads per iteration: isolates whether the gain comes
     * from instruction-level parallelism (multiple NoC reads in flight) vs just a
     * sequential / bursted address stream. Each path advances by `ilp` lines. */
    uint64_t t0 = rdcycle();
    if (ilp == 1) {
        for (uint64_t i = 0; i < nlines; i += 1) {
            uint64_t p0 = base + i * 64;
            uint64_t acc;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vmv.x.s %0, v0\n"
                : "=r"(acc)
                : "r"(p0)
                : "memory", "v0");
            sink ^= acc;
        }
    } else if (ilp == 2) {
        for (uint64_t i = 0; i < nlines; i += 2) {
            uint64_t b = base + i * 64;
            uint64_t p0 = b, p1 = b + 64;
            uint64_t acc;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vle64.v v1, (%2)\n"
                "vxor.vv v0, v0, v1\n"
                "vmv.x.s %0, v0\n"
                : "=r"(acc)
                : "r"(p0), "r"(p1)
                : "memory", "v0", "v1");
            sink ^= acc;
        }
    } else if (ilp == 4) {
        for (uint64_t i = 0; i < nlines; i += 4) {
            uint64_t b = base + i * 64;
            uint64_t p0 = b, p1 = b + 64, p2 = b + 128, p3 = b + 192;
            uint64_t acc;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vle64.v v1, (%2)\n"
                "vle64.v v2, (%3)\n"
                "vle64.v v3, (%4)\n"
                "vxor.vv v0, v0, v1\n"
                "vxor.vv v2, v2, v3\n"
                "vxor.vv v0, v0, v2\n"
                "vmv.x.s %0, v0\n"
                : "=r"(acc)
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3)
                : "memory", "v0", "v1", "v2", "v3");
            sink ^= acc;
        }
    } else { /* ilp == 8 */
        for (uint64_t i = 0; i < nlines; i += 8) {
            uint64_t b = base + i * 64;
            uint64_t p0 = b, p1 = b + 64, p2 = b + 128, p3 = b + 192;
            uint64_t p4 = b + 256, p5 = b + 320, p6 = b + 384, p7 = b + 448;
            uint64_t acc;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vle64.v v1, (%2)\n"
                "vle64.v v2, (%3)\n"
                "vle64.v v3, (%4)\n"
                "vle64.v v4, (%5)\n"
                "vle64.v v5, (%6)\n"
                "vle64.v v6, (%7)\n"
                "vle64.v v7, (%8)\n"
                "vxor.vv v0, v0, v1\n"
                "vxor.vv v2, v2, v3\n"
                "vxor.vv v4, v4, v5\n"
                "vxor.vv v6, v6, v7\n"
                "vxor.vv v0, v0, v2\n"
                "vxor.vv v4, v4, v6\n"
                "vxor.vv v0, v0, v4\n"
                "vmv.x.s %0, v0\n"
                : "=r"(acc)
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6), "r"(p7)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
            sink ^= acc;
        }
    }
    uint64_t t1 = rdcycle();

    *(volatile uint64_t*)(slot + RES_CYCLES) = (t1 - t0);
    *(volatile uint64_t*)(slot + RES_BYTES) = nlines * 64ULL;
    *(volatile uint64_t*)(slot + RES_SINK) = sink;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

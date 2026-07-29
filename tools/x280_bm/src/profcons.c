/*
 * profcons.c - X280 TRUE CONSUMER for the SPSC kernel profiler (FAST variant).
 *
 * Drains every per-RISC L1 ring (kernel_profiler.hpp producers block-on-full),
 * advances each ring head so producers unblock, and relays the drained markers to
 * host pinned memory (D2H). This version uses the fastest read setup we found
 * (gridilp / pollmp): **`nharts` harts** each draining a disjoint slice of cores,
 * reading the ring's bulk data as **64 B `vle64` flits with 4 independent reads in
 * flight (ILP 4)** and relaying via posted `vse64` writes. 2 harts x ILP 4 reached
 * ~1.5 GB/s; 4 harts deep-ILP collapses, so nharts is capped at 3 (default 2).
 *
 * SPSC contract (per Tensix core, per RISC r in 0..4) inside profiler_msg_t @
 * prof_l1 (control_vector[32] then buffer[r] at +128 + r*2048):
 *   tail (producer) = ctrl[DEVICE_BUFFER_END_INDEX_BR_ER + r] = ctrl[5+r]
 *   head (consumer) = ctrl[HOST_BUFFER_END_INDEX_BR_ER   + r] = ctrl[r]
 *   ring storage    = buffer[r].data[0..RING_CAP-1]  (RING_CAP = 512 words)
 *   tail/head MONOTONIC word counts; storage index = count % RING_CAP.
 *
 * Each hart uses its own PCIe write window (WRITE_WIN_BASE + hartid) + one
 * read/write window per core in its slice (index = core index). Host relay layout:
 * slice (c*5+r) at host_base + (c*5+r)*slice_words*4 (drained words appended,
 * clamped to slice_words; head still advances past it so flow control never stalls).
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1
 *                          +0x18 num_cores +0x20 slice_words +0x28 stop
 *                          +0x30 nonce +0x38 nharts
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 total_words +0x08 u64 loops
 *                          +0x10 u64 max_outstanding +0x18 u64 done (= DONE_MAGIC)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   HEADS   @ 0x08013000 : num_cores*5 x u32 (consumer head per ring, LIM-local)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define HEADS_BASE 0x08013000UL

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_NUM_CORES (MBOX_PARAMS + 0x18)
#define P_SLICE_WORDS (MBOX_PARAMS + 0x20)
#define P_STOP (MBOX_PARAMS + 0x28)
#define P_NONCE (MBOX_PARAMS + 0x30)
#define P_NHARTS (MBOX_PARAMS + 0x38)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_TOTAL 0x00
#define RES_LOOPS 0x08
#define RES_MAXOUT 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0xC0570FFEE1ULL
#define FOOTER_MAGIC 0xC05D09F11E12345ULL

#define NRISC 5
#define RING_CAP 512u
#define WRITE_WIN_BASE 200u
#define CTRL_HEAD(r) (r)
#define CTRL_TAIL(r) (5u + (r))

/* Isolated-throughput bench config (host-set), separate from the params block. */
#define BENCH_CFG 0x08011600UL
#define B_WORDS (BENCH_CFG + 0x00) /* words drained per ring per pass (0 = normal mode) */
#define B_REPS (BENCH_CFG + 0x08)  /* number of drain passes */
#define B_MODE (BENCH_CFG + 0x10)  /* bit0: 1 = read-only (skip relay), 0 = read+relay */

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
static inline uint32_t r32(uint64_t a) { return *(volatile uint32_t*)a; }
static inline void w32(uint64_t a, uint32_t v) { *(volatile uint32_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }

/* Read 4 independent 64 B flits from the ring (ILP 4) and posted-write them out. */
static inline void copy4_flits(uint64_t rp, uint64_t wp) {
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vle64.v v0, (%0)\n"
        "vle64.v v1, (%1)\n"
        "vle64.v v2, (%2)\n"
        "vle64.v v3, (%3)\n"
        "vse64.v v0, (%4)\n"
        "vse64.v v1, (%5)\n"
        "vse64.v v2, (%6)\n"
        "vse64.v v3, (%7)\n"
        :
        : "r"(rp), "r"(rp + 64), "r"(rp + 128), "r"(rp + 192), "r"(wp), "r"(wp + 64), "r"(wp + 128), "r"(wp + 192)
        : "memory", "v0", "v1", "v2", "v3");
}
static inline void copy1_flit(uint64_t rp, uint64_t wp) {
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vle64.v v0, (%0)\n"
        "vse64.v v0, (%1)\n"
        :
        : "r"(rp), "r"(wp)
        : "memory", "v0");
}
/* read-only variants (bench diagnostic): 4 / 1 flits in flight, no relay write */
static inline void read4_flits(uint64_t rp) {
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vle64.v v0, (%0)\n"
        "vle64.v v1, (%1)\n"
        "vle64.v v2, (%2)\n"
        "vle64.v v3, (%3)\n"
        :
        : "r"(rp), "r"(rp + 64), "r"(rp + 128), "r"(rp + 192)
        : "memory", "v0", "v1", "v2", "v3");
}
static inline void read1_flit(uint64_t rp) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vle64.v v0, (%0)\n" : : "r"(rp) : "memory", "v0");
}

int main(uint64_t hartid) {
    uint64_t host_base = r64(P_HOST_BASE);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    uint64_t slice_words = r64(P_SLICE_WORDS);
    uint64_t nharts = r64(P_NHARTS);
    uint64_t pcie_enc = r64(P_PCIE_ENC);
    (void)r64(P_NONCE);
    if (nharts == 0 || nharts > 3) {
        nharts = 2;
    }

    /* every hart clears its own result slot */
    volatile uint64_t* rl = (volatile uint64_t*)RES_SLOT(hartid);
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    fence_();

    if (hartid >= nharts) {
        fence_();
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    /* this hart's contiguous slice of cores */
    uint64_t q = (num_cores + nharts - 1) / nharts;
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }

    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    volatile uint32_t* heads = (volatile uint32_t*)HEADS_BASE;
    for (uint64_t c = lo; c < hi; c++) {
        for (uint32_t r = 0; r < NRISC; r++) {
            heads[c * NRISC + r] = 0;
        }
    }

    uint64_t pcie_x = pcie_enc & 0x3f;
    uint64_t pcie_y = (pcie_enc >> 6) & 0x3f;
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t off_w = host_base & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint32_t write_win = WRITE_WIN_BASE + (uint32_t)hartid;

    /* this hart's write window -> PCIe tile, host IOVA, POSTED */
    noc_tlb_2m_t wt;
    wt.data[0] = 0;
    wt.data[1] = 0;
    wt.data[2] = 0;
    wt.data[3] = 0;
    wt.addr = host_base >> 21;
    wt.x_end = (uint32_t)pcie_x;
    wt.y_end = (uint32_t)pcie_y;
    wt.x_start = (uint32_t)pcie_x;
    wt.y_start = (uint32_t)pcie_y;
    wt.posted = 1;
    wt.noc_selector = 0;
    (void)noc_configure_tlb_2m_ext(write_win, &wt, 0);

    /* one read/write window per core in this hart's slice (index = core index) */
    for (uint64_t c = lo; c < hi; c++) {
        (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], prof_l1, 0, 0);
    }
    fence_();

    uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)write_win * NOC_2M_WINDOW_STRIDE + off_w;

    /* --- isolated throughput benchmark: drain `bench_words`/ring for `bench_reps`
     * passes with no producers, timed; measures the consumer's peak read+relay rate.
     * (bench_words <= slice_words and <= RING_CAP so head=0 needs no wrap/clamp.) --- */
    uint64_t bench_words = r64(B_WORDS);
    uint64_t bench_reps = r64(B_REPS);
    uint64_t bench_ro = r64(B_MODE) & 1ULL; /* 1 = read-only (no relay) */
    if (bench_words) {
        uint64_t bytes = 0;
        uint64_t t0 = rdcycle();
        for (uint64_t rep = 0; rep < bench_reps; rep++) {
            for (uint64_t c = lo; c < hi; c++) {
                uint64_t rbufs = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                for (uint32_t r = 0; r < NRISC; r++) {
                    uint64_t rp = rbufs + (uint64_t)r * 2048;
                    uint64_t wp = wbase + (uint64_t)(c * NRISC + r) * slice_words * 4;
                    uint32_t nflits = (uint32_t)(bench_words / 8);
                    uint32_t fi = 0;
                    if (bench_ro) {
                        for (; fi + 4 <= nflits; fi += 4) {
                            read4_flits(rp + (uint64_t)fi * 64);
                        }
                        for (; fi < nflits; fi++) {
                            read1_flit(rp + (uint64_t)fi * 64);
                        }
                        for (uint32_t ww = nflits * 8; ww < bench_words; ww++) {
                            (void)r32(rp + (uint64_t)ww * 4);
                        }
                    } else {
                        for (; fi + 4 <= nflits; fi += 4) {
                            copy4_flits(rp + (uint64_t)fi * 64, wp + (uint64_t)fi * 64);
                        }
                        for (; fi < nflits; fi++) {
                            copy1_flit(rp + (uint64_t)fi * 64, wp + (uint64_t)fi * 64);
                        }
                        for (uint32_t ww = nflits * 8; ww < bench_words; ww++) {
                            w32(wp + (uint64_t)ww * 4, r32(rp + (uint64_t)ww * 4));
                        }
                    }
                    bytes += bench_words * 4;
                }
            }
        }
        uint64_t t1 = rdcycle();
        fence_();
        w64(RES_SLOT(hartid) + RES_TOTAL, bytes);
        w64(RES_SLOT(hartid) + RES_LOOPS, t1 - t0); /* cycles */
        fence_();
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        fence_();
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint64_t total = 0, loops = 0, max_out = 0;

    for (;;) {
        uint64_t progressed = 0;
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
            uint64_t rbufs = cbase + 128;
            for (uint32_t r = 0; r < NRISC; r++) {
                uint32_t tail = r32(cbase + CTRL_TAIL(r) * 4);
                uint32_t head = heads[c * NRISC + r];
                if (tail == head) {
                    continue;
                }
                if ((uint64_t)(tail - head) > max_out) {
                    max_out = tail - head;
                }
                uint64_t ring_base = rbufs + (uint64_t)r * 2048;
                uint64_t slice_base = wbase + (uint64_t)(c * NRISC + r) * slice_words * 4;

                uint32_t h2 = head;
                while (h2 != tail) {
                    uint32_t start = h2 % RING_CAP;
                    uint32_t run = tail - h2;
                    if (run > RING_CAP - start) {
                        run = RING_CAP - start; /* contiguous slab, no wrap */
                    }
                    uint64_t rp = ring_base + (uint64_t)start * 4;
                    if ((uint64_t)h2 + run <= slice_words) {
                        /* fast path: whole slab lands in the host slice */
                        uint64_t wp = slice_base + (uint64_t)h2 * 4;
                        uint32_t nflits = run / 8;
                        uint32_t fi = 0;
                        for (; fi + 4 <= nflits; fi += 4) {
                            copy4_flits(rp + (uint64_t)fi * 64, wp + (uint64_t)fi * 64);
                        }
                        for (; fi < nflits; fi++) {
                            copy1_flit(rp + (uint64_t)fi * 64, wp + (uint64_t)fi * 64);
                        }
                        for (uint32_t ww = nflits * 8; ww < run; ww++) {
                            w32(wp + (uint64_t)ww * 4, r32(rp + (uint64_t)ww * 4));
                        }
                    } else {
                        /* slow path: slab exceeds slice -> scalar copy with clamp */
                        for (uint32_t ww = 0; ww < run; ww++) {
                            uint32_t val = r32(rp + (uint64_t)ww * 4);
                            if ((uint64_t)h2 + ww < slice_words) {
                                w32(slice_base + (uint64_t)(h2 + ww) * 4, val);
                            }
                        }
                    }
                    h2 += run;
                    heads[c * NRISC + r] = h2;
                    w32(cbase + CTRL_HEAD(r) * 4, h2); /* advance head so producer unblocks */
                }
                total += (uint64_t)(tail - head);
                progressed = 1;
            }
        }
        loops++;
        if (r64(P_STOP) && !progressed) {
            break;
        }
    }

    /* per-hart footer (final posted write) so the host knows the relay landed */
    fence_();
    uint64_t footer_region = num_cores * NRISC * slice_words * 4ULL;
    w64(wbase + footer_region + hartid * 64, FOOTER_MAGIC);
    fence_();

    w64(RES_SLOT(hartid) + RES_TOTAL, total);
    w64(RES_SLOT(hartid) + RES_LOOPS, loops);
    w64(RES_SLOT(hartid) + RES_MAXOUT, max_out);
    fence_();
    w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
    fence_();
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

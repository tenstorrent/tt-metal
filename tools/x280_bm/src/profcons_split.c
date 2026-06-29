/*
 * profcons_split.c - X280 profiler consumer, READER/RELAY-HART SPLIT (bench).
 *
 * Decouples the device-read from the host-relay by putting them on SEPARATE harts,
 * communicating through a per-reader LIM staging ring (an on-X280 SPSC):
 *   - reader harts (0..nread-1): drain device rings (System Port, NOC0) with
 *     contiguous ILP-4 reads and stage each ring's 64 flits CONTIGUOUSLY in LIM,
 *     plus one descriptor (host dst_start) per ring.
 *   - relay hart (last): per ring, read the descriptor once and BATCH-copy all 64
 *     flits (4 KB) to host as one contiguous run (NOC1), via wide RVV (e64,m8).
 * Reads never stall on a write; the relay's posted writes overlap the readers'
 * reads on a different instruction stream. Batching amortizes the per-flit dst read
 * + bookkeeping (64 -> 1 per ring) and turns scattered 64 B writes into 4 KB
 * contiguous posted bursts (the d2hbw-style regime).
 *
 * Bench only (uniform 64-flit rings, `reps` passes); the relay hart times the whole
 * concurrent window. The read-only modes (P_MODE 1/2) isolate the read path.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1
 *                          +0x18 num_cores +0x20 nharts +0x28 nread +0x30 reps
 *                          +0x38 mode
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40: +0x00 bytes +0x08 cycles
 *                          +0x18 done (= DONE_MAGIC)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   STAGECTL@ 0x08018000 : per reader h: +h*32 prod, +8 cons, +16 rdone (flit counts)
 *   STAGE   @ 0x08020000 : reader h at +h*STAGE_STRIDE (264 KiB):
 *                          +0x00000 FLIT ring: NREC=4096 x 64 B flits (= 64 rings buffered)
 *                          +0x40000 DESC ring: NDESC=128 x u64 host dst_start (per ring)
 *
 * Host pre-zeros STAGECTL before boot (no init race). Host slice for (core c,risc r)
 * = host_base + (c*5+r)*2048; region = num_cores*5*2048 < 2 MiB.
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define STAGECTL 0x08018000UL
#define STAGE_BASE 0x08020000UL
/* X280 LIM is 1.875 MiB; staging lives above the 64 KiB managed region. 264 KiB per
 * reader (256 KiB flit ring + descriptors); nread<=4 -> <=1.03 MiB, fits comfortably. */
#define STAGE_STRIDE 0x42000UL

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_NUM_CORES (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_NREAD (MBOX_PARAMS + 0x28)
#define P_REPS (MBOX_PARAMS + 0x30)
#define P_MODE (MBOX_PARAMS + 0x38) /* 0 = full read->stage->relay; 1 = read-only scatter; 2 = read-only contiguous */

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_BYTES 0x00
#define RES_CYCLES 0x08
#define RES_DONE 0x18
#define DONE_MAGIC 0x5717C0FFEEULL

#define NRISC 5
#define FLITS_PER_RING 64u /* 64 x 64 B flits drained per (core,risc) */
#define NREC 4096u         /* flit-ring depth per reader (= 64 rings buffered) */
#define NDESC 128u         /* descriptor-ring depth (> rings-in-flight, no overwrite) */
#define HOST_SLICE 2048u   /* host stride per (core,risc) */
#define WRITE_WIN_BASE 200u

#define PROD(h) (STAGECTL + (uint64_t)(h) * 32 + 0)
#define CONS(h) (STAGECTL + (uint64_t)(h) * 32 + 8)
#define RDONE(h) (STAGECTL + (uint64_t)(h) * 32 + 16)
#define FBASE(h) (STAGE_BASE + (uint64_t)(h) * STAGE_STRIDE)                       /* flit ring */
#define DBASE(h) (STAGE_BASE + (uint64_t)(h) * STAGE_STRIDE + (uint64_t)NREC * 64) /* desc ring (after flits) */
#define FADDR(h, i) (FBASE(h) + ((uint64_t)(i) % NREC) * 64)                       /* flit slot */
#define DADDR(h, k) (DBASE(h) + ((uint64_t)(k) % NDESC) * 8)                       /* desc slot */

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
/* device read one 64 B flit into v0 */
static inline void vread(uint64_t rp) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vle64.v v0, (%0)\n" : : "r"(rp) : "memory", "v0");
}
/* store v0 (64 B) to addr */
static inline void vwrite(uint64_t wp) { __asm__ volatile("vse64.v v0, (%0)\n" : : "r"(wp) : "memory"); }
/* ILP-4 reader primitive: 4 independent device reads in flight, THEN store the 4
 * flits to LIM staging (w0..w3). Hides the read latency across 4 reads (pollmp). */
static inline void read4_store4(uint64_t rp, uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {
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
        : "r"(rp), "r"(rp + 64), "r"(rp + 128), "r"(rp + 192), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
        : "memory", "v0", "v1", "v2", "v3");
}
/* TRUE-SCATTER ILP-4 (read-only A/B bench only): 4 reads from 4 DIFFERENT cores.
 * Measured a wash vs read4_store4 -- the System Port already overlaps same-endpoint
 * reads -- so the production reader uses read4_store4. Kept for the --ro bench. */
static inline void read4_store4_scatter(
    uint64_t rp0, uint64_t rp1, uint64_t rp2, uint64_t rp3, uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {
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
        : "r"(rp0), "r"(rp1), "r"(rp2), "r"(rp3), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
        : "memory", "v0", "v1", "v2", "v3");
}
/* BATCHED relay copy: move `nelem` contiguous e64 (8 B) elements from LIM src to
 * host dst as a contiguous posted burst. Wide vector group (m8) minimizes
 * instruction count; 2-way ILP (two independent m8 streams, v0-7 and v8-15) issues
 * two LIM loads before either store so the second load hides the first's latency.
 * LIM reads are local/fast; host writes are posted (non-stalling). */
static inline void copy_contig(uint64_t src, uint64_t dst, uint64_t nelem) {
    uint64_t i = 0;
    uint64_t vl;
    __asm__ volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(nelem));
    while (i + 2 * vl <= nelem) {
        __asm__ volatile(
            "vle64.v v0, (%0)\n"
            "vle64.v v8, (%1)\n"
            "vse64.v v0, (%2)\n"
            "vse64.v v8, (%3)\n"
            :
            : "r"(src + i * 8), "r"(src + (i + vl) * 8), "r"(dst + i * 8), "r"(dst + (i + vl) * 8)
            : "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v8",
              "v9",
              "v10",
              "v11",
              "v12",
              "v13",
              "v14",
              "v15");
        i += 2 * vl;
    }
    while (i < nelem) {
        uint64_t vl2;
        __asm__ volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl2) : "r"(nelem - i));
        __asm__ volatile("vle64.v v0, (%0)\n vse64.v v0, (%1)\n"
                         :
                         : "r"(src + i * 8), "r"(dst + i * 8)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        i += vl2;
    }
}

int main(uint64_t hartid) {
    uint64_t pcie_enc = r64(P_PCIE_ENC);
    uint64_t host_base = r64(P_HOST_BASE);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    uint64_t nharts = r64(P_NHARTS);
    uint64_t nread = r64(P_NREAD);
    uint64_t reps = r64(P_REPS);
    uint64_t mode = r64(P_MODE);

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

    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t off_w = host_base & (NOC_2M_WINDOW_STRIDE - 1ULL);

    if (hartid < nread) {
        /* ---------------- READER: contiguous ILP-4 read -> contiguous flit stage ---------------- */
        uint64_t q = (num_cores + nread - 1) / nread;
        uint64_t lo = hartid * q, hi = lo + q;
        if (hi > num_cores) {
            hi = num_cores;
        }
        for (uint64_t c = lo; c < hi; c++) {
            (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], prof_l1, 0, 0);
        }
        fence_();

        if (mode != 0) {
            /* ===== READ-ONLY timing (A/B bench): isolate the read rate, no relay.
             * mode 1 = quarter-scatter (4 distinct cores per ILP-4 group), mode 2 =
             * contiguous (4 flits at one core). Flits land in a fixed LIM scratch. */
            uint64_t half = hi - lo, qq = half / 4, rem = half % 4;
            uint64_t start[5];
            start[0] = lo;
            for (int j = 0; j < 4; j++) {
                start[j + 1] = start[j] + qq + ((uint64_t)j < rem ? 1 : 0);
            }
            uint64_t w0 = FBASE(hartid), w1 = w0 + 128, w2 = w0 + 256, w3 = w0 + 384;
            uint64_t bytes = 0;
            uint64_t t0 = rdcycle();
            for (uint64_t rep = 0; rep < reps; rep++) {
                if (mode == 1) {
                    for (uint64_t i = 0; i < qq; i++) {
                        uint64_t c0 = start[0] + i, c1 = start[1] + i, c2 = start[2] + i, c3 = start[3] + i;
                        uint64_t b0 = NOC_2M_WINDOW_BASE + c0 * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        uint64_t b1 = NOC_2M_WINDOW_BASE + c1 * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        uint64_t b2 = NOC_2M_WINDOW_BASE + c2 * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        uint64_t b3 = NOC_2M_WINDOW_BASE + c3 * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        for (uint32_t r = 0; r < NRISC; r++) {
                            uint64_t rp0 = b0 + (uint64_t)r * 2048, rp1 = b1 + (uint64_t)r * 2048;
                            uint64_t rp2 = b2 + (uint64_t)r * 2048, rp3 = b3 + (uint64_t)r * 2048;
                            for (uint32_t f = 0; f < FLITS_PER_RING; f++) {
                                read4_store4_scatter(
                                    rp0 + (uint64_t)f * 64,
                                    rp1 + (uint64_t)f * 64,
                                    rp2 + (uint64_t)f * 64,
                                    rp3 + (uint64_t)f * 64,
                                    w0,
                                    w1,
                                    w2,
                                    w3);
                                bytes += 256;
                            }
                        }
                    }
                    for (uint64_t j = 0; j < rem; j++) {
                        uint64_t c = start[j] + qq;
                        uint64_t rbufs = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        for (uint32_t r = 0; r < NRISC; r++) {
                            uint64_t rp = rbufs + (uint64_t)r * 2048;
                            for (uint32_t g = 0; g < FLITS_PER_RING; g += 4) {
                                read4_store4(rp + (uint64_t)g * 64, w0, w1, w2, w3);
                                bytes += 256;
                            }
                        }
                    }
                } else {
                    for (uint64_t c = lo; c < hi; c++) {
                        uint64_t rbufs = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                        for (uint32_t r = 0; r < NRISC; r++) {
                            uint64_t rp = rbufs + (uint64_t)r * 2048;
                            for (uint32_t g = 0; g < FLITS_PER_RING; g += 4) {
                                read4_store4(rp + (uint64_t)g * 64, w0, w1, w2, w3);
                                bytes += 256;
                            }
                        }
                    }
                }
            }
            uint64_t t1 = rdcycle();
            w64(RES_SLOT(hartid) + RES_BYTES, bytes);
            w64(RES_SLOT(hartid) + RES_CYCLES, t1 - t0);
            fence_();
            w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
            for (;;) {
                __asm__ volatile("wfi");
            }
        }

        uint32_t prod = 0; /* flit count (monotonic; ring-aligned to FLITS_PER_RING) */
        uint64_t bytes = 0;
        for (uint64_t rep = 0; rep < reps; rep++) {
            for (uint64_t c = lo; c < hi; c++) {
                uint64_t rbufs = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                for (uint32_t r = 0; r < NRISC; r++) {
                    uint64_t rp = rbufs + (uint64_t)r * 2048;
                    uint64_t dst0 = (uint64_t)(c * NRISC + r) * HOST_SLICE;
                    /* wait for room for a whole ring, publish its descriptor, then
                     * stage 64 flits contiguously (16 ILP-4 groups). */
                    while ((prod + FLITS_PER_RING - r32(CONS(hartid))) > NREC) {
                        /* flit ring full: wait for relay to drain a ring */
                    }
                    w64(DADDR(hartid, prod / FLITS_PER_RING), dst0);
                    uint64_t fp = FADDR(hartid, prod);
                    for (uint32_t g = 0; g < FLITS_PER_RING; g += 4) {
                        uint64_t w = fp + (uint64_t)g * 64;
                        read4_store4(rp + (uint64_t)g * 64, w, w + 64, w + 128, w + 192);
                        bytes += 256;
                    }
                    prod += FLITS_PER_RING;
                    fence_(); /* publish ring: flits+desc visible before PROD advances */
                    w32(PROD(hartid), prod);
                }
            }
        }
        fence_();
        w64(RDONE(hartid), 1);
        w64(RES_SLOT(hartid) + RES_BYTES, bytes);
        fence_();
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        for (;;) {
            __asm__ volatile("wfi");
        }
    } else {
        /* ---------------- RELAY: per-ring batched copy, PARTITIONED ----------------
         * Each relay hart drains a DISJOINT subset of readers [lo_r,hi_r) -- with
         * nrelay==nread this is a 1:1 reader<->relay pairing = two fully independent
         * pipelines (own reader, own LIM buffer, own NOC1 write window, own host
         * slice). No shared ring => no SPMC race (unlike the old all-relays-sweep-
         * all-readers attempt). With nrelay==1 the lone relay drains every reader. */
        if (mode != 0) { /* read-only bench: no relay work */
            fence_();
            w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
            for (;;) {
                __asm__ volatile("wfi");
            }
        }
        uint64_t nrelay = nharts - nread;
        uint64_t r_idx = hartid - nread;
        uint64_t lo_r = r_idx * nread / nrelay;
        uint64_t hi_r = (r_idx + 1) * nread / nrelay;
        uint64_t pcie_x = pcie_enc & 0x3f;
        uint64_t pcie_y = (pcie_enc >> 6) & 0x3f;
        uint32_t write_win = WRITE_WIN_BASE + (uint32_t)hartid;
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
        wt.noc_selector = 1; /* NOC1: split relay writes off the readers' NOC0 reads */
        (void)noc_configure_tlb_2m_ext(write_win, &wt, 0);
        fence_();
        uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)write_win * NOC_2M_WINDOW_STRIDE + off_w;

        uint32_t cons[4] = {0, 0, 0, 0};
        uint64_t bytes = 0;
        uint64_t t0 = rdcycle();
        for (;;) {
            uint64_t progressed = 0, all_done = 1;
            for (uint64_t h = lo_r; h < hi_r; h++) {
                uint32_t pr = r32(PROD(h)); /* flit count, advances per ring (x64) */
                uint32_t cn = cons[h];
                /* drain whole rings: one descriptor read + one 4 KB contiguous copy each */
                while ((uint32_t)(pr - cn) >= FLITS_PER_RING) {
                    uint64_t dst0 = r64(DADDR(h, cn / FLITS_PER_RING));
                    copy_contig(FADDR(h, cn), wbase + dst0, (uint64_t)FLITS_PER_RING * 8);
                    cn += FLITS_PER_RING;
                    bytes += (uint64_t)FLITS_PER_RING * 64;
                    progressed = 1;
                }
                /* tail (< one ring; not expected in bench) -- per-flit via descriptor */
                while (pr != cn) {
                    uint64_t dst0 = r64(DADDR(h, cn / FLITS_PER_RING));
                    uint64_t off = (uint64_t)(cn % FLITS_PER_RING) * 64;
                    vread(FADDR(h, cn));
                    vwrite(wbase + dst0 + off);
                    cn++;
                    bytes += 64;
                    progressed = 1;
                }
                cons[h] = cn;
                w32(CONS(h), cn);
                if (!r64(RDONE(h))) {
                    all_done = 0;
                }
            }
            if (all_done && !progressed) {
                break;
            }
        }
        uint64_t t1 = rdcycle();
        /* footer = final posted write to host (just past the relay region) so the host
         * can confirm NOC1 -> PCIe -> host actually landed (not silently dropped). */
        fence_();
        w64(wbase + (uint64_t)num_cores * NRISC * HOST_SLICE, 0xF00DD2C0FFEEULL);
        fence_();
        w64(RES_SLOT(hartid) + RES_BYTES, bytes);
        w64(RES_SLOT(hartid) + RES_CYCLES, t1 - t0);
        fence_();
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    return 0;
}

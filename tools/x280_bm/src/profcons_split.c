/*
 * profcons_split.c - X280 profiler consumer, READER/RELAY-HART SPLIT (bench).
 *
 * Decouples the device-read from the host-relay by putting them on SEPARATE harts,
 * communicating through a per-reader LIM staging ring (an on-X280 SPSC):
 *   - reader harts (0..nread-1): drain device rings (System Port, NOC0) and push
 *     each 64 B flit + its host destination into their LIM staging ring.
 *   - relay hart (last): pop staged records and posted-write the flits to host.
 * Reads never stall on a write; the relay's posted writes overlap the readers'
 * reads on a different instruction stream. This isolates the "separate relay hart"
 * win (target: recover the ~748 MB/s read-only rate vs 327 fused). Reader read is
 * still SEQUENTIAL here; true-scatter ILP-4 is the next step.
 *
 * Bench only (uniform 512-word rings, `reps` passes), to measure end-to-end
 * read->stage->relay throughput; the relay hart times the whole concurrent window.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1
 *                          +0x18 num_cores +0x20 nharts +0x28 nread +0x30 reps
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40: +0x00 bytes +0x08 cycles
 *                          +0x18 done (= DONE_MAGIC)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   STAGECTL@ 0x08018000 : per reader h: +h*32 prod, +8 cons, +16 rdone (u64 each)
 *   STAGE   @ 0x08020000 : reader h ring at +h*0x10000, NREC=512 x 128B records:
 *                          [0..7] dst_byte_off, [64..127] flit (64 B)
 *
 * Host pre-zeros STAGECTL before boot (no init race). Host slice for (core c,risc r)
 * = host_base + (c*5+r)*2048 (one ring = 2 KB); region = num_cores*5*2048 < 2 MiB.
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define STAGECTL 0x08018000UL
#define STAGE_BASE 0x08020000UL
#define STAGE_STRIDE 0x10000UL /* 64 KiB per reader ring */

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_NUM_CORES (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_NREAD (MBOX_PARAMS + 0x28)
#define P_REPS (MBOX_PARAMS + 0x30)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_BYTES 0x00
#define RES_CYCLES 0x08
#define RES_DONE 0x18
#define DONE_MAGIC 0x5717C0FFEEULL

#define NRISC 5
#define FLITS_PER_RING 64u /* 512 words / 8 */
#define NREC 512u
#define REC_SZ 128u
#define HOST_SLICE 2048u /* bytes per (core,risc) in host = one ring */
#define WRITE_WIN_BASE 200u

#define PROD(h) (STAGECTL + (uint64_t)(h) * 32 + 0)
#define CONS(h) (STAGECTL + (uint64_t)(h) * 32 + 8)
#define RDONE(h) (STAGECTL + (uint64_t)(h) * 32 + 16)
#define REC(h, i) (STAGE_BASE + (uint64_t)(h) * STAGE_STRIDE + ((uint64_t)(i) % NREC) * REC_SZ)

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
/* relay ILP-4: load 4 staged flits from LIM (s0..s3), posted-write to 4 host dsts (d0..d3) */
static inline void lim4_to_host4(
    uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
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
        : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
        : "memory", "v0", "v1", "v2", "v3");
}

int main(uint64_t hartid) {
    uint64_t pcie_enc = r64(P_PCIE_ENC);
    uint64_t host_base = r64(P_HOST_BASE);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    uint64_t nharts = r64(P_NHARTS);
    uint64_t nread = r64(P_NREAD);
    uint64_t reps = r64(P_REPS);

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
        /* ---------------- READER ---------------- */
        uint64_t q = (num_cores + nread - 1) / nread;
        uint64_t lo = hartid * q, hi = lo + q;
        if (hi > num_cores) {
            hi = num_cores;
        }
        for (uint64_t c = lo; c < hi; c++) {
            (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], prof_l1, 0, 0);
        }
        fence_();

        uint32_t prod = 0;
        uint64_t bytes = 0;
        for (uint64_t rep = 0; rep < reps; rep++) {
            for (uint64_t c = lo; c < hi; c++) {
                uint64_t rbufs = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off + 128;
                for (uint32_t r = 0; r < NRISC; r++) {
                    uint64_t rp = rbufs + (uint64_t)r * 2048;
                    uint64_t dst0 = (uint64_t)(c * NRISC + r) * HOST_SLICE;
                    /* 64 flits = 16 groups of 4; ILP-4 read then 4 LIM stores per group */
                    for (uint32_t g = 0; g < FLITS_PER_RING; g += 4) {
                        while ((prod + 4 - r32(CONS(hartid))) > NREC) {
                            /* staging ring full: wait for relay to drain */
                        }
                        uint64_t r0 = REC(hartid, prod), r1 = REC(hartid, prod + 1);
                        uint64_t r2 = REC(hartid, prod + 2), r3 = REC(hartid, prod + 3);
                        w64(r0 + 0, dst0 + (uint64_t)(g + 0) * 64); /* host dst headers */
                        w64(r1 + 0, dst0 + (uint64_t)(g + 1) * 64);
                        w64(r2 + 0, dst0 + (uint64_t)(g + 2) * 64);
                        w64(r3 + 0, dst0 + (uint64_t)(g + 3) * 64);
                        read4_store4(rp + (uint64_t)g * 64, r0 + 64, r1 + 64, r2 + 64, r3 + 64);
                        prod += 4;
                        bytes += 256;
                    }
                    /* publish this ring's records once (one fence per ring, not per flit) */
                    fence_();
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
        /* ---------------- RELAY ---------------- */
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
            for (uint64_t h = 0; h < nread; h++) {
                uint32_t pr = r32(PROD(h)); /* read producer index once per sweep */
                uint32_t cn = cons[h];
                while (pr - cn >= 4) { /* ILP-4: 4 LIM reads -> 4 posted host writes in flight */
                    uint64_t a0 = REC(h, cn), a1 = REC(h, cn + 1), a2 = REC(h, cn + 2), a3 = REC(h, cn + 3);
                    lim4_to_host4(
                        a0 + 64,
                        a1 + 64,
                        a2 + 64,
                        a3 + 64,
                        wbase + r64(a0),
                        wbase + r64(a1),
                        wbase + r64(a2),
                        wbase + r64(a3));
                    cn += 4;
                    bytes += 256;
                    progressed = 1;
                }
                while (pr != cn) { /* remainder (1..3) */
                    uint64_t rec = REC(h, cn);
                    uint64_t dst = r64(rec + 0);
                    vread(rec + 64);
                    vwrite(wbase + dst);
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

/*
 * poll6n1.c - X280 NIU-vs-mesh decider: 4 harts vector-poll on NOC0 + 2 DMA
 * channels on NOC1, concurrently, for a fixed duration. This is the one config
 * that puts real read demand on BOTH NIUs at once (the harts saturate the NOC0
 * path at ~530; the DMA adds 2 streams on the NOC1 NIU). If total exceeds 530 the
 * limit was a single NIU; if it stays ~530 the wall is the shared mesh ingress
 * into the L2CPU tile (downstream of both NIUs).
 *
 * Harts poll NOC0 (noc_selector=0); the DMA channels' NoC TLB windows are
 * programmed with noc_selector=1 (NOC1), same translated coord (proven by
 * noc1_probe). Fixed-duration peak-throughput test; total bytes / time.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 dst_base
 *                          +0x18 duration_cycles +0x20 nharts +0x28 dma_on
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 load_reads +0x08 u64 dma_reads +0x10 u64 done
 *   COORDS  @ 0x08011200 : num_cores x { u32 x, u32 y } (translated)
 */
#include <stdint.h>

#include "dma_engine.h" /* pulls in noc.h */

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL

#define P_NUM_CORES (MBOX_PARAMS + 0x00)
#define P_SRC_L1 (MBOX_PARAMS + 0x08)
#define P_DST_BASE (MBOX_PARAMS + 0x10)
#define P_DURATION (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_DMA_ON (MBOX_PARAMS + 0x28)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_LOAD 0x00
#define RES_DMA 0x08
#define RES_DONE 0x10
#define DONE_MAGIC 0x6217E6C1DEULL

#define DMA_CH_BASE(n) (X280_DMA_CH0 + (uint64_t)(n) * 0x58ULL)
#define CH_SAR 0x00
#define CH_DAR 0x08
#define CH_LLP 0x10
#define CH_CTL 0x18
#define CH_CFG 0x40
#define DMA_WIN0 110u
#define DMA_WIN1 111u

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }

static uint64_t g_ctl_base;
static void compute_ctl_base(void) {
    x280_dma_ctl0_t c;
    c.val = 0;
    c.f.int_en = 1;
    c.f.done = 1;
    c.f.tt_fc = 3;
    c.f.src_msize = 2;
    c.f.dest_msize = 2;
    c.f.src_tr_width = 5;
    c.f.dst_tr_width = 5;
    c.f.sms = 0;
    c.f.dms = 1;
    g_ctl_base = c.val;
}
static void chan_init(int n) {
    uint64_t b = DMA_CH_BASE(n);
    w64(b + CH_CTL, g_ctl_base);
    x280_dma_cfg0_t g;
    g.val = 0;
    g.f.ch_prior = 1;
    g.f.hs_sel_src = 1;
    g.f.hs_sel_dst = 1;
    w64(b + CH_CFG, g.val);
    w64(b + CH_LLP, 1);
    uint64_t mb = (1ULL << n) | (1ULL << (8 + n));
    w64(X280_DMA_INT + X280_DMA_INT_MASKTFR, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKBLK, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKSRCT, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKDSTT, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKERR, mb);
}
static void chan_kick(int n, uint64_t sar, uint64_t dar, uint32_t words32) {
    uint64_t b = DMA_CH_BASE(n);
    x280_dma_ctl0_t c;
    c.val = g_ctl_base;
    c.f.block_ts = words32;
    w64(b + CH_CTL, c.val);
    w64(b + CH_SAR, sar);
    w64(b + CH_DAR, dar);
    uint64_t cbit = (1ULL << n);
    uint64_t ebit = (1ULL << n) | (1ULL << (8 + n));
    w64(X280_DMA_INT + X280_DMA_INT_CLEARTFR, cbit);
    w64(X280_DMA_INT + X280_DMA_INT_CLEARERR, cbit);
    w64(X280_DMA_INT + X280_DMA_INT_CLEARDSTT, cbit);
    w64(X280_DMA_INT + X280_DMA_INT_CLEARSRCT, cbit);
    w64(X280_DMA_INT + X280_DMA_INT_CLEARBLK, cbit);
    fence_();
    w64(X280_DMA_MISC + X280_DMA_MISC_CHEN, ebit);
    fence_();
    w64(X280_DMA_SWHS + X280_DMA_SWHS_LSTSRC, ebit);
    w64(X280_DMA_SWHS + X280_DMA_SWHS_LSTDST, ebit);
    w64(X280_DMA_SWHS + X280_DMA_SWHS_REQSRC, ebit);
    w64(X280_DMA_SWHS + X280_DMA_SWHS_REQDST, ebit);
    w64(X280_DMA_SWHS + X280_DMA_SWHS_SGLRQSRC, ebit);
    w64(X280_DMA_SWHS + X280_DMA_SWHS_SGLRQDST, ebit);
    fence_();
}
static inline int chan_done(int n) { return (r64(X280_DMA_INT + X280_DMA_INT_RAWTFR) & (1ULL << n)) ? 1 : 0; }
static inline uint64_t chan_sar(uint32_t win, uint64_t off) {
    return X280_DMA_NOC_BASE + (uint64_t)win * NOC_2M_WINDOW_STRIDE + off;
}
/* Program a NoC TLB window on the chosen NoC (0/1). */
static void cfg_win_sel(uint32_t idx, uint32_t x, uint32_t y, uint64_t addr, int sel) {
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
    tlb.posted = 1;
    tlb.noc_selector = (uint64_t)sel;
    (void)noc_configure_tlb_2m_ext(idx, &tlb, 0);
}

int main(uint64_t hartid) {
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t duration = *(volatile uint64_t*)P_DURATION;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t dma_on = *(volatile uint64_t*)P_DMA_ON;
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    if (duration == 0) {
        duration = 50000000ULL;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t slot = RES_SLOT(hartid);

    if (hartid >= nharts) {
        w64(slot + RES_LOAD, 0);
        w64(slot + RES_DMA, 0);
        fence_();
        w64(slot + RES_DONE, DONE_MAGIC);
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
    /* Harts poll NOC0. */
    for (uint64_t c = lo; c < hi; c++) {
        (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], src_l1, 0, 0);
    }

    int drive_dma = (dma_on && hartid == 0);
    uint64_t dma_reads = 0;
    uint64_t dma_dst0 = dst_base + num_cores * 8;
    uint64_t dma_dst1 = dma_dst0 + 64;
    if (drive_dma) {
        compute_ctl_base();
        dma_engine_init();
        chan_init(0);
        chan_init(1);
        /* DMA windows on NOC1 (sel=1), re-reading cores 0/1. */
        cfg_win_sel(DMA_WIN0, coords[0], coords[1], src_l1, 1);
        cfg_win_sel(DMA_WIN1, coords[2], coords[3], src_l1, 1);
        fence_();
        chan_kick(0, chan_sar(DMA_WIN0, off), dma_dst0, 2);
        chan_kick(1, chan_sar(DMA_WIN1, off), dma_dst1, 2);
    }

    w64(slot + RES_LOAD, 0);
    w64(slot + RES_DMA, 0);
    fence_();

    uint64_t load_reads = 0, sink = 0, c = lo;
    uint64_t t0 = rdcycle();
    while ((rdcycle() - t0) < duration) {
        if (drive_dma) {
            if (chan_done(0)) {
                dma_reads++;
                chan_kick(0, chan_sar(DMA_WIN0, off), dma_dst0, 2);
            }
            if (chan_done(1)) {
                dma_reads++;
                chan_kick(1, chan_sar(DMA_WIN1, off), dma_dst1, 2);
            }
        }
        if (hi > lo) {
            uint64_t ptr = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off;
            uint64_t a0;
            __asm__ volatile(
                "vsetivli zero, 8, e64, m1, ta, ma\n"
                "vle64.v v0, (%1)\n"
                "vmv.x.s %0, v0\n"
                : "=r"(a0)
                : "r"(ptr)
                : "memory", "v0");
            sink ^= a0;
            *(volatile uint64_t*)(dst_base + c * 8) = a0;
            c++;
            if (c >= hi) {
                c = lo;
            }
            load_reads++;
        }
    }

    w64(slot + RES_LOAD, load_reads);
    w64(slot + RES_DMA, dma_reads);
    (void)sink;
    fence_();
    w64(slot + RES_DONE, DONE_MAGIC);
    fence_();
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

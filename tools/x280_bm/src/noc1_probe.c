/*
 * noc1_probe.c - Determine how the X280 NoC TLB reaches a Tensix tile on NOC1.
 *
 * The 2 MiB TLB descriptor has a noc_selector bit (0=NOC0, 1=NOC1). We confirmed
 * NOC0 reads use the TRANSLATED coord. For NOC1 the destination coord might be
 * the SAME translated coord (routing handles it) or the tile's NOC1 coord. A
 * wrong coord on a blocking load would hang the in-order hart, so we probe via
 * the DMA (its wait has a cycle timeout -> reports rc=2 instead of hanging).
 *
 * Three reads of core 0's flit:
 *   A: sel=0 + translated coord  (ground truth -- this is the proven NOC0 path)
 *   B: sel=1 + translated coord  (does the translated coord work on NOC1?)
 *   C: sel=1 + NOC1 coord        (does the explicit NOC1 coord work?)
 * Whichever of B/C returns rc=0 and value == A is the correct NOC1 config.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 u32 trans_x +0x04 u32 trans_y
 *                          +0x08 u32 noc1_x  +0x0C u32 noc1_y  +0x10 u64 src_l1
 *   RESULTS @ 0x08011040 : +0x00 u32 rcA +0x08 u64 valA
 *                          +0x10 u32 rcB +0x18 u64 valB
 *                          +0x20 u32 rcC +0x28 u64 valC  +0x30 u64 done
 */
#include <stdint.h>

#include "dma_engine.h" /* pulls in noc.h */

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define DST_SCRATCH 0x08012000UL
#define DONE_MAGIC 0x10C1C0FFEEULL

#define DMA_CH_BASE(n) (X280_DMA_CH0 + (uint64_t)(n) * 0x58ULL)
#define CH_SAR 0x00
#define CH_DAR 0x08
#define CH_LLP 0x10
#define CH_CTL 0x18
#define CH_CFG 0x40
#define PROBE_WIN 50u /* arbitrary unused TLB window for the probe */

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
static int chan_wait(int n, uint64_t timeout) {
    uint64_t t0 = rdcycle();
    uint64_t m = (1ULL << n);
    for (;;) {
        if (r64(X280_DMA_INT + X280_DMA_INT_RAWERR) & m) {
            return 1;
        }
        if (r64(X280_DMA_INT + X280_DMA_INT_RAWTFR) & m) {
            return 0;
        }
        if (rdcycle() - t0 > timeout) {
            return 2;
        }
    }
}

/* Program the probe window to (x,y,addr) on the chosen NoC, then DMA-read one
 * 64 B flit into scratch. Returns rc (0 ok / 2 timeout); *out = value read. */
static int probe(uint32_t x, uint32_t y, uint64_t addr, int sel, uint64_t* out) {
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
    (void)noc_configure_tlb_2m_ext(PROBE_WIN, &tlb, 0);
    (void)*(volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)PROBE_WIN * 0x10);
    fence_();
    uint64_t off = addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t sar = X280_DMA_NOC_BASE + (uint64_t)PROBE_WIN * NOC_2M_WINDOW_STRIDE + off;
    *(volatile uint64_t*)DST_SCRATCH = 0; /* clear scratch */
    fence_();
    chan_kick(0, sar, DST_SCRATCH, 2 /*64B*/);
    int rc = chan_wait(0, X280_DMA_TIMEOUT_CYCLES);
    *out = *(volatile uint64_t*)DST_SCRATCH;
    return rc;
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    uint32_t tx = *(volatile uint32_t*)(MBOX_PARAMS + 0x00);
    uint32_t ty = *(volatile uint32_t*)(MBOX_PARAMS + 0x04);
    uint32_t n1x = *(volatile uint32_t*)(MBOX_PARAMS + 0x08);
    uint32_t n1y = *(volatile uint32_t*)(MBOX_PARAMS + 0x0C);
    uint64_t src_l1 = *(volatile uint64_t*)(MBOX_PARAMS + 0x10);

    volatile uint64_t* rl = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    fence_();

    compute_ctl_base();
    dma_engine_init();
    chan_init(0);

    uint64_t vA = 0, vB = 0, vC = 0;
    int rcA = probe(tx, ty, src_l1, 0, &vA);
    int rcB = probe(tx, ty, src_l1, 1, &vB);
    int rcC = probe(n1x, n1y, src_l1, 1, &vC);

    w64(MBOX_RESULTS + 0x00, (uint64_t)(uint32_t)rcA);
    w64(MBOX_RESULTS + 0x08, vA);
    w64(MBOX_RESULTS + 0x10, (uint64_t)(uint32_t)rcB);
    w64(MBOX_RESULTS + 0x18, vB);
    w64(MBOX_RESULTS + 0x20, (uint64_t)(uint32_t)rcC);
    w64(MBOX_RESULTS + 0x28, vC);
    fence_();
    w64(MBOX_RESULTS + 0x30, DONE_MAGIC);
    fence_();
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

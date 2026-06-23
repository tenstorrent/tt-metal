/*
 * grid_drain4.c - X280 multi-channel grid drain: split the grid into sections,
 * one DMA channel per section, draining them CONCURRENTLY.
 *
 * Runs on hart 0. The Synopsys DW DMAC at 0x2FF80000 has channel register blocks
 * at CH0 + N*0x58; the global SW-handshake / interrupt / channel-enable registers
 * carry one bit per channel (bit N, write-enable bit 8+N). Each channel uses its
 * own CPU NOC TLB index N (and the matching SAR window) so 4 channels can target
 * 4 different cores at once, all routed through the single outbound DMA TLB slot 8.
 *
 * Because the reference only ever used channel 0 and the channel count here is
 * unconfirmed, the FW first SELF-TESTS channels 0..nchan_req-1 (one solo transfer
 * each; the DMA wait has a cycle timeout, so a missing channel reports rc=2 rather
 * than hanging). The working set becomes `chan_mask`; the grid is split into that
 * many contiguous sections and drained concurrently.
 *
 * Per-core transfer is a single 32B-word burst block: bytes must be <= 256
 * (burst 8 x 32B) and a multiple of 32. 256 B = 4 flits = the design point.
 *
 * LIM layout (must match the host):
 *   PARAMS  @ 0x08011000 : +0x00 num_cores +0x08 src_l1 +0x10 bytes
 *                          +0x18 dst_base  +0x20 nrounds +0x28 nchan_req
 *   RESULTS @ 0x08011040 : +0x00 rc +0x08 total_cycles +0x10 u32 rounds
 *                          +0x14 u32 cores +0x18 done +0x20 round_min
 *                          +0x28 u32 c0_r0 +0x2C u32 c0_rlast
 *                          +0x30 u32 chan_mask +0x34 u32 nsec +0x38 u32 fail_core
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (virtual coords)
 *   DST     @ dst_base    : num_cores x bytes
 */
#include <stdint.h>

#include "dma_engine.h" /* pulls in noc.h */

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL

#define P_NUM_CORES (MBOX_PARAMS + 0x00)
#define P_SRC_L1 (MBOX_PARAMS + 0x08)
#define P_BYTES (MBOX_PARAMS + 0x10)
#define P_DST_BASE (MBOX_PARAMS + 0x18)
#define P_NROUNDS (MBOX_PARAMS + 0x20)
#define P_NCHAN (MBOX_PARAMS + 0x28)

#define R_RC (MBOX_RESULTS + 0x00)
#define R_TOTAL (MBOX_RESULTS + 0x08)
#define R_ROUNDS (MBOX_RESULTS + 0x10)
#define R_CORES (MBOX_RESULTS + 0x14)
#define R_DONE (MBOX_RESULTS + 0x18)
#define R_ROUND_MIN (MBOX_RESULTS + 0x20)
#define R_C0_R0 (MBOX_RESULTS + 0x28)
#define R_C0_RLAST (MBOX_RESULTS + 0x2C)
#define R_CHAN_MASK (MBOX_RESULTS + 0x30)
#define R_NSEC (MBOX_RESULTS + 0x34)
#define R_FAIL_CORE (MBOX_RESULTS + 0x38)

#define DONE_MAGIC 0x6D1DD4A1E4ULL

/* DW DMAC per-channel register block stride + offsets (relative to channel base). */
#define DMA_CH_STRIDE 0x58ULL
#define DMA_CH_BASE(n) (X280_DMA_CH0 + (uint64_t)(n) * DMA_CH_STRIDE)
#define CH_SAR 0x00
#define CH_DAR 0x08
#define CH_LLP 0x10
#define CH_CTL 0x18
#define CH_CFG 0x40

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }

/* Base CTL value for a NOC(EXTERN)->LIM(L2), 32B-word, burst-8, sw-handshake
 * transfer (block_ts OR'd in per kick). Matches dma_engine.h do_dma's channel-0
 * setup. */
static uint64_t g_ctl_base;
static void compute_ctl_base(void) {
    x280_dma_ctl0_t c;
    c.val = 0;
    c.f.int_en = 1;
    c.f.done = 1;
    c.f.tt_fc = 3;     /* PERIPH<->PERIPH */
    c.f.src_msize = 2; /* burst 8 */
    c.f.dest_msize = 2;
    c.f.src_tr_width = 5; /* 32B word */
    c.f.dst_tr_width = 5;
    c.f.sms = 0; /* EXTERN src master (NOC) */
    c.f.dms = 1; /* L2 dst master (LIM) */
    g_ctl_base = c.val;
}

/* One-time per-channel setup: CTL/CFG/LLP + unmask channel n. */
static void chan_init(int n) {
    uint64_t b = DMA_CH_BASE(n);
    w64(b + CH_CTL, g_ctl_base);
    x280_dma_cfg0_t g;
    g.val = 0;
    g.f.ch_prior = 1;
    g.f.hs_sel_src = 1;
    g.f.hs_sel_dst = 1;
    w64(b + CH_CFG, g.val);
    w64(b + CH_LLP, 1);                            /* lms = L2, loc = 0 (llp disabled in CTL) */
    uint64_t mb = (1ULL << n) | (1ULL << (8 + n)); /* mask bit + write-enable */
    w64(X280_DMA_INT + X280_DMA_INT_MASKTFR, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKBLK, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKSRCT, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKDSTT, mb);
    w64(X280_DMA_INT + X280_DMA_INT_MASKERR, mb);
}

/* Kick a single-burst block on channel n (SAR/DAR set, channel enabled, handshake
 * asserted). 256 B = 8 x 32B = one burst of msize 8, so a single handshake moves
 * the whole block. */
static void chan_kick(int n, uint64_t sar, uint64_t dar, uint32_t words32) {
    uint64_t b = DMA_CH_BASE(n);
    x280_dma_ctl0_t c;
    c.val = g_ctl_base;
    c.f.block_ts = words32;
    w64(b + CH_CTL, c.val);
    w64(b + CH_SAR, sar);
    w64(b + CH_DAR, dar);
    uint64_t cbit = (1ULL << n);                     /* clear: bit only */
    uint64_t ebit = (1ULL << n) | (1ULL << (8 + n)); /* enable/req: bit + we */
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

/* Wait for channel n's block-transfer-complete (RAWTFR bit n). 0 ok / 1 err / 2 timeout. */
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

/* DMA-side SAR for channel n's CPU NOC TLB window (index n). */
static inline uint64_t chan_sar(int n, uint64_t noc_addr) {
    uint64_t off = noc_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return X280_DMA_NOC_BASE + (uint64_t)n * NOC_2M_WINDOW_STRIDE + off;
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t src_l1 = *(volatile uint64_t*)P_SRC_L1;
    uint64_t bytes = *(volatile uint64_t*)P_BYTES;
    uint64_t dst_base = *(volatile uint64_t*)P_DST_BASE;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nchan_req = *(volatile uint64_t*)P_NCHAN;
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (nchan_req == 0 || nchan_req > 4) {
        nchan_req = 4;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint32_t words32 = (uint32_t)(bytes / 32);

    /* Prime results line + zero destination. */
    volatile uint64_t* rline = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rline[i] = 0;
    }
    volatile uint64_t* d64 = (volatile uint64_t*)dst_base;
    for (uint64_t i = 0; i < (num_cores * bytes + 7) / 8; i++) {
        d64[i] = 0;
    }
    fence_();

    compute_ctl_base();
    dma_engine_init(); /* outbound DMA TLB slot 8 = 0x200 (shared by all channels) */
    for (uint64_t n = 0; n < nchan_req; n++) {
        chan_init((int)n);
    }

    /* ---- Self-test channels 0..nchan_req-1 (solo transfer from core 0) ---- */
    uint32_t chan_mask = 0;
    for (uint64_t n = 0; n < nchan_req; n++) {
        (void)noc_configure_tlb_2m((uint32_t)n, coords[0], coords[1], src_l1, 1, 0);
        (void)*(volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + n * 0x10); /* readback fence */
        fence_();
        chan_kick((int)n, chan_sar((int)n, src_l1), dst_base + n * bytes, words32);
        if (chan_wait((int)n, X280_DMA_TIMEOUT_CYCLES) == 0) {
            chan_mask |= (1u << n);
        }
    }

    /* Build the working-channel list. */
    int working[4];
    uint32_t nsec = 0;
    for (int n = 0; n < 4; n++) {
        if (chan_mask & (1u << n)) {
            working[nsec++] = n;
        }
    }

    int rc = (nsec == 0) ? 2 : 0;
    uint32_t fail_core = 0xFFFFFFFFu;
    uint64_t total = 0, round_min = ~0ULL;
    uint32_t c0_r0 = 0, c0_rlast = 0;
    uint64_t r = 0;

    if (nsec > 0) {
        uint64_t secsz = (num_cores + nsec - 1) / nsec; /* ceil */
        uint64_t off = src_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
        for (r = 0; r < nrounds && rc == 0; r++) {
            uint64_t a = rdcycle();
            for (uint64_t step = 0; step < secsz && rc == 0; step++) {
                int act_ch[4];
                uint64_t act_core[4];
                int nact = 0;
                /* Program one core per active section's channel. */
                for (uint32_t ci = 0; ci < nsec; ci++) {
                    uint64_t core_idx = (uint64_t)ci * secsz + step;
                    uint64_t sec_end = (uint64_t)(ci + 1) * secsz;
                    if (sec_end > num_cores) {
                        sec_end = num_cores;
                    }
                    if (core_idx >= sec_end) {
                        continue; /* this section exhausted */
                    }
                    int ch = working[ci];
                    (void)noc_configure_tlb_2m(
                        (uint32_t)ch, coords[core_idx * 2], coords[core_idx * 2 + 1], src_l1, 1, 0);
                    act_ch[nact] = ch;
                    act_core[nact] = core_idx;
                    nact++;
                }
                /* Drain the peripheral-port write pipeline for all programmed TLBs. */
                for (int k = 0; k < nact; k++) {
                    (void)*(volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)act_ch[k] * 0x10);
                }
                fence_();
                /* Kick all active channels (now all in flight). */
                for (int k = 0; k < nact; k++) {
                    chan_kick(
                        act_ch[k],
                        chan_sar(act_ch[k], src_l1) /*=base+ch*stride+off*/,
                        dst_base + act_core[k] * bytes,
                        words32);
                }
                (void)off;
                /* Wait for all active channels. */
                for (int k = 0; k < nact; k++) {
                    int w = chan_wait(act_ch[k], X280_DMA_TIMEOUT_CYCLES);
                    if (w != 0) {
                        rc = w;
                        fail_core = (uint32_t)act_core[k];
                        break;
                    }
                }
            }
            uint64_t b = rdcycle();
            uint64_t round_cyc = b - a;
            total += round_cyc;
            if (round_cyc < round_min) {
                round_min = round_cyc;
            }
            uint32_t c0 = *(volatile uint32_t*)dst_base;
            if (r == 0) {
                c0_r0 = c0;
            }
            c0_rlast = c0;
        }
    }

    *(volatile uint64_t*)R_RC = (uint64_t)rc;
    *(volatile uint64_t*)R_TOTAL = total;
    *(volatile uint32_t*)R_ROUNDS = (uint32_t)r;
    *(volatile uint32_t*)R_CORES = (uint32_t)num_cores;
    *(volatile uint64_t*)R_ROUND_MIN = (round_min == ~0ULL) ? 0 : round_min;
    *(volatile uint32_t*)R_C0_R0 = c0_r0;
    *(volatile uint32_t*)R_C0_RLAST = c0_rlast;
    *(volatile uint32_t*)R_CHAN_MASK = chan_mask;
    *(volatile uint32_t*)R_NSEC = nsec;
    *(volatile uint32_t*)R_FAIL_CORE = fail_core;
    fence_();
    *(volatile uint64_t*)R_DONE = DONE_MAGIC;
    fence_();

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

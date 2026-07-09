/*
 * grid_drain.c - X280 bare-metal grid drain: DMA-pull N flits from every core.
 *
 * Runs on hart 0 (others park). Reads params + a coordinate table from LIM
 * (host-written before release), then repeatedly drains the worker grid: each
 * ROUND walks all `num_cores` Tensix cores and DMAs `bytes_per_core` (e.g. 256 B
 * = 4 NoC flits) from each core's L1 into a per-core slot in X280 LIM, with no
 * core loads. One DMA channel, serial over cores.
 *
 * Uses the setup-once result: the channel/CTL/CFG/word-size + interrupt mask are
 * programmed ONCE (first dma_engine_noc_to_x280 to core 0); per-core work is then
 * just reprogram-the-NoC-TLB-coord + dma_retrigger_block (restore block_ts/SAR/
 * DAR, kick, wait). bytes_per_core must be a multiple of 32 (single 32B-word
 * block) -- 256 B fits.
 *
 * LIM layout (must match the host example):
 *   PARAMS  @ 0x08011000 : +0x00 u64 num_cores  +0x08 u64 src_l1
 *                          +0x10 u64 bytes_per_core  +0x18 u64 dst_base
 *                          +0x20 u64 nrounds
 *   RESULTS @ 0x08011040 : +0x00 u64 rc          +0x08 u64 total_cycles (all rounds)
 *                          +0x10 u32 rounds_done +0x14 u32 cores
 *                          +0x18 u64 done (= DONE_MAGIC, written last)
 *                          +0x20 u64 round_min_cycles
 *                          +0x28 u32 c0_round0   +0x2C u32 c0_roundlast (core0 counter)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y }
 *   DST     @ dst_base    : num_cores x bytes_per_core
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

#define R_RC (MBOX_RESULTS + 0x00)
#define R_TOTAL (MBOX_RESULTS + 0x08)
#define R_ROUNDS (MBOX_RESULTS + 0x10)
#define R_CORES (MBOX_RESULTS + 0x14)
#define R_DONE (MBOX_RESULTS + 0x18)
#define R_ROUND_MIN (MBOX_RESULTS + 0x20)
#define R_C0_R0 (MBOX_RESULTS + 0x28)
#define R_C0_RLAST (MBOX_RESULTS + 0x2C)
#define R_FAIL_CORE (MBOX_RESULTS + 0x30) /* u32 (0xFFFFFFFF = none) */
#define R_FAIL_X (MBOX_RESULTS + 0x34)    /* u32 */
#define R_FAIL_Y (MBOX_RESULTS + 0x38)    /* u32 */

#define DONE_MAGIC 0x6D1DD4A1E0ULL /* "GRID DRAIN" */

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* Lightweight re-trigger of a single 32B-word block (CTL/CFG/word-size/masters/
 * interrupt-mask already set by the first dma_engine_noc_to_x280). */
static int dma_retrigger_block(uint64_t src_dma_sar, uint64_t dst, uint32_t words32) {
    x280_dma_set_block_ts(words32);
    x280_dma_set_sar(src_dma_sar);
    x280_dma_set_dar(dst);
    x280_dma_clear_interrupts();
    x280_dma_fence_();
    x280_dma_channel_enable();
    x280_dma_fence_();
    x280_dma_start_burst();
    x280_dma_fence_();
    return x280_dma_wait_done(X280_DMA_TIMEOUT_CYCLES);
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
    if (nrounds == 0) {
        nrounds = 1;
    }
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS; /* [x0,y0,x1,y1,...] */

    /* Prime results line (ECC). */
    volatile uint64_t* rline = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rline[i] = 0;
    }

    /* Zero the whole destination region (ECC + clean baseline). */
    volatile uint64_t* d64 = (volatile uint64_t*)dst_base;
    uint64_t total_words = (num_cores * bytes + 7) / 8;
    for (uint64_t i = 0; i < total_words; i++) {
        d64[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    uint32_t words32 = (uint32_t)(bytes / 32);
    uint64_t src_dma_sar = dma_engine_noc_sar_dar(src_l1);

    /* One-time full setup (channel/CTL/CFG/word-size/mask), via a first move
     * from core 0. */
    dma_engine_init();
    int rc = dma_engine_noc_to_x280(coords[0], coords[1], src_l1, X280_DMA_MASTER_L2, dst_base, (uint32_t)bytes);

    uint64_t total = 0, round_min = ~0ULL;
    uint32_t c0_r0 = 0, c0_rlast = 0;
    uint32_t fail_core = 0xFFFFFFFFu, fail_x = 0, fail_y = 0;
    uint64_t r;
    for (r = 0; r < nrounds && rc == 0; r++) {
        uint64_t a = rdcycle();
        for (uint64_t i = 0; i < num_cores; i++) {
            uint32_t x = coords[i * 2 + 0];
            uint32_t y = coords[i * 2 + 1];
            dma_engine_program_noc_tlb(x, y, src_l1);
            rc = dma_retrigger_block(src_dma_sar, dst_base + i * bytes, words32);
            if (rc != 0) {
                fail_core = (uint32_t)i;
                fail_x = x;
                fail_y = y;
                break;
            }
        }
        uint64_t b = rdcycle();
        uint64_t round_cyc = b - a;
        total += round_cyc;
        if (round_cyc < round_min) {
            round_min = round_cyc;
        }
        uint32_t c0 = *(volatile uint32_t*)dst_base; /* core0 counter this round */
        if (r == 0) {
            c0_r0 = c0;
        }
        c0_rlast = c0;
    }

    *(volatile uint64_t*)R_RC = (uint64_t)rc;
    *(volatile uint64_t*)R_TOTAL = total;
    *(volatile uint32_t*)R_ROUNDS = (uint32_t)r;
    *(volatile uint32_t*)R_CORES = (uint32_t)num_cores;
    *(volatile uint64_t*)R_ROUND_MIN = (round_min == ~0ULL) ? 0 : round_min;
    *(volatile uint32_t*)R_C0_R0 = c0_r0;
    *(volatile uint32_t*)R_C0_RLAST = c0_rlast;
    *(volatile uint32_t*)R_FAIL_CORE = fail_core;
    *(volatile uint32_t*)R_FAIL_X = fail_x;
    *(volatile uint32_t*)R_FAIL_Y = fail_y;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)R_DONE = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// x280 TLB-window store BANDWIDTH firmware. Measures how fast the x280 can push
// data out over the NOC through a TLB window — the go/no-go number for hosting
// the fabric mux here (the mux's whole job is streaming packet payloads out).
//
// Method: aim window 0 at this same tile's uncached GDDR (NOC loopback, so the
// test touches no other core), then time tight loops of stores through the
// window aperture with the RISC-V cycle counter (mcycle). Core runs at the
// 200 MHz PLL solution set by the boot tool, so MB/s = bytes * 200e6 / cycles.
//
// Results land in the mailbox result area for the boot tool to read:
//   0x301000F0 : bw_done flag (0x00B7D09E when the sweep has finished)
//   0x30100100 : N records, each {u32 tag, u32 size_bytes, u64 cycles}
//     tag 0..3 = posted-ordering 64-bit stores, sizes 4K/16K/64K/256K
//     tag 4    = default-ordering 64-bit stores, 64K (ordering cost)
//     tag 5    = posted-ordering 32-bit stores,  64K (store-width cost)

#include <stdint.h>

#define MBOX 0x30100000UL
#define MBOX_BW_DONE (MBOX + 0xF0)
#define MBOX_RESULTS (MBOX + 0x100)
#define REG32(a) (*(volatile uint32_t*)(uintptr_t)(a))
#define REG64(a) (*(volatile uint64_t*)(uintptr_t)(a))

#define TLB_CFG_BASE 0x20000000UL
#define TLB_APERTURE 0x0430000000UL  // small window 0, uncached aperture
#define WINDOW_MASK 0x1FFFFFUL

// Target a Tensix core's L1 (NOC0 1,2 = logical worker 0,0), the same
// known-good destination the echo firmware used. Do NOT loop back to this tile
// (8,3): a NOC write from a tile to itself deadlocks the tile's NOC, which wedges
// the whole L2CPU endpoint (reads then time out to all-0xFF). L1 offset 0x40000
// is scratch; the sweep tops out at 256 KiB (0x40000..0x80000, within 1.5 MiB L1).
#define TGT_X 1
#define TGT_Y 2
#define TGT_ADDR 0x40000UL

#define FW_STATE_ALIVE 0xA11FEull
#define BW_DONE 0x00B7D09Eu

static inline void fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }

static inline uint64_t rd_mcycle(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

// ordering: 0 = default, 2 = posted writes (noc_properties_lo bits 25:26).
static void set_window(uint32_t x, uint32_t y, uint32_t addr, uint32_t ordering) {
    volatile uint64_t* cfg64 = (volatile uint64_t*)(uintptr_t)TLB_CFG_BASE;
    volatile uint32_t* cfg32 = (volatile uint32_t*)(uintptr_t)TLB_CFG_BASE;
    fence();
    cfg64[0] = (uint64_t)addr >> 21;
    cfg32[2] = (x & 0x3f) | ((y & 0x3f) << 6) | ((ordering & 0x3) << 25);
    cfg32[3] = 0;
    fence();
}

static inline uint64_t store64_loop(uint32_t size) {
    volatile uint64_t* ap = (volatile uint64_t*)(uintptr_t)(TLB_APERTURE + (TGT_ADDR & WINDOW_MASK));
    uint64_t n = size / 8;
    uint64_t pat = 0x1122334455667788ull;
    fence();
    uint64_t c0 = rd_mcycle();
    for (uint64_t i = 0; i < n; i++) {
        ap[i] = pat + i;
    }
    fence();
    uint64_t c1 = rd_mcycle();
    return c1 - c0;
}

static inline uint64_t store32_loop(uint32_t size) {
    volatile uint32_t* ap = (volatile uint32_t*)(uintptr_t)(TLB_APERTURE + (TGT_ADDR & WINDOW_MASK));
    uint64_t n = size / 4;
    uint32_t pat = 0xA5A50000u;
    fence();
    uint64_t c0 = rd_mcycle();
    for (uint64_t i = 0; i < n; i++) {
        ap[i] = pat + (uint32_t)i;
    }
    fence();
    uint64_t c1 = rd_mcycle();
    return c1 - c0;
}

static void record(uint32_t idx, uint32_t tag, uint32_t size, uint64_t cycles) {
    volatile uint32_t* r = (volatile uint32_t*)(uintptr_t)(MBOX_RESULTS + idx * 16);
    r[0] = tag;
    r[1] = size;
    r[2] = (uint32_t)cycles;
    r[3] = (uint32_t)(cycles >> 32);
    fence();
}

void fw_main(void) {
    // Enable the cycle counter (clear mcountinhibit; traps-and-skips if absent).
    __asm__ volatile("csrw mcountinhibit, zero" ::: "memory");
    REG64(MBOX + 0x08) = FW_STATE_ALIVE;

    const uint32_t sizes[4] = {4096, 16384, 65536, 262144};

    // Headline sweep: posted-ordering 64-bit stores.
    set_window(TGT_X, TGT_Y, TGT_ADDR, 2);
    // One warm-up pass (prime the window / any first-touch cost).
    (void)store64_loop(4096);
    for (uint32_t i = 0; i < 4; i++) {
        record(i, i, sizes[i], store64_loop(sizes[i]));
    }

    // Comparison A: default (non-posted) ordering, 64 KiB.
    set_window(TGT_X, TGT_Y, TGT_ADDR, 0);
    (void)store64_loop(4096);
    record(4, 4, 65536, store64_loop(65536));

    // Comparison B: posted ordering, 32-bit stores, 64 KiB.
    set_window(TGT_X, TGT_Y, TGT_ADDR, 2);
    (void)store32_loop(4096);
    record(5, 5, 65536, store32_loop(65536));

    REG32(MBOX_BW_DONE) = BW_DONE;
    fence();

    // Park with a heartbeat so the hart stays alive (persistent; never exits).
    uint64_t hb = 0;
    for (;;) {
        REG64(MBOX + 0x00) = ++hb;
        fence();
    }
}

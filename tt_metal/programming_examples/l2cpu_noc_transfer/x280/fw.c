// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// x280 "echo" firmware: proof that an L2CPU hart can actively communicate back
// to a Tensix core over the NOC.
//
// Runs from the tile's 2 MiB LIM (L3-as-scratchpad) at 0x0800_0000. Loops on a
// mailbox in LIM; when a Tensix kernel posts a request {seq, x, y, addr, value},
// the firmware aims TLB window 0 at (x, y), probe-READS addr expecting a magic
// word (reads are harmless if the coordinate encoding is wrong), and only on a
// match window-WRITES value*3+7 into addr+16 — which a Tensix kernel is polling.
//
// The x280 has no NOC command interface: the window config (dest x/y in
// noc_properties) plus plain loads/stores through the window aperture are the
// only way it can touch the NOC. This file is the whole "noc-lite" story.
//
// Cache strategy: mailbox lines are cbo.flush'd/cbo.inval'd each pass. If the
// core lacks Zicbom the trap handler (start.S) skips the opcode and we rely on
// front-port coherence; trap_count/cmo_ok in the mailbox report which happened.
//
// Mailbox layout (LIM + 0x0010_0000), 64 B cache-line ownership:
//   line 0 (fw-owned):   0x00 heartbeat u64 | 0x08 fw_state u64 | 0x10 hartid u64
//                        0x18 trap_count u64 | 0x20 mcause_last u64 | 0x28 cmo_ok u64
//                        0x38 trap-handler scratch
//   line 1 (fw-owned):   0x40 resp_seq u32 | 0x44 status u32 (1=ok, 2=probe mismatch)
//                        0x48 probe_value u32 | 0x4c result u32
//   line 2 (tensix-owned): 0x80 req_seq u32 | 0x84 x u32 | 0x88 y u32
//                        0x8c addr u32 | 0x90 value u32

#include <stdint.h>

// Mailbox in the UNCACHED GDDR alias so the firmware's stores land in physical
// DRAM immediately and a Tensix NOC read sees them without any cache flush.
// (Firmware .text runs from the cached GDDR alias 0x4000_3000_0000 per dram.ld.)
#define MBOX 0x30100000UL
#define REG64(a) (*(volatile uint64_t*)(uintptr_t)(a))
#define REG32(a) (*(volatile uint32_t*)(uintptr_t)(a))

// Small TLB window 0: config registers + uncached access aperture.
#define TLB_CFG_BASE 0x20000000UL
#define TLB_APERTURE 0x0430000000UL
#define WINDOW_MASK 0x1FFFFFUL

#define MAGIC 0x5AFE0001u
#define FW_STATE_ALIVE 0xA11FEull

static inline void fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }

static inline void cbo_flush(uintptr_t a) { __asm__ volatile("cbo.flush (%0)" ::"r"(a) : "memory"); }

static inline void cbo_inval(uintptr_t a) { __asm__ volatile("cbo.inval (%0)" ::"r"(a) : "memory"); }

static inline uint64_t read_mhartid(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mhartid" : "=r"(v));
    return v;
}

// Point small window 0 at NOC0 tile (x, y); the low 21 bits of the target
// address come from the offset within the 2 MiB aperture, the rest from
// local_offset. noc_properties_lo: x_end[5:0], y_end[11:6]; everything else
// (multicast, ordering=default, VC, noc_sel=NOC0) left zero.
static void set_window(uint32_t x, uint32_t y, uint32_t addr) {
    volatile uint64_t* cfg64 = (volatile uint64_t*)(uintptr_t)TLB_CFG_BASE;
    volatile uint32_t* cfg32 = (volatile uint32_t*)(uintptr_t)TLB_CFG_BASE;
    fence();
    cfg64[0] = (uint64_t)addr >> 21;
    cfg32[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    cfg32[3] = 0;
    fence();
}

void fw_main(void) {
    uint64_t hb = 0;

    REG64(MBOX + 0x10) = read_mhartid();

    // Probe Zicbom support: if cbo.flush traps, the handler bumps trap_count.
    uint64_t traps_before = REG64(MBOX + 0x18);
    cbo_flush(MBOX);
    REG64(MBOX + 0x28) = (REG64(MBOX + 0x18) == traps_before) ? 1 : 0;

    // A request already in the mailbox at boot is stale — ignore it.
    uint32_t last_seq = REG32(MBOX + 0x80);

    REG64(MBOX + 0x08) = FW_STATE_ALIVE;

    for (;;) {
        REG64(MBOX + 0x00) = ++hb;
        cbo_flush(MBOX + 0x00);
        fence();

        cbo_inval(MBOX + 0x80);
        fence();
        uint32_t seq = REG32(MBOX + 0x80);
        if (seq != 0 && seq != last_seq) {
            uint32_t x = REG32(MBOX + 0x84);
            uint32_t y = REG32(MBOX + 0x88);
            uint32_t addr = REG32(MBOX + 0x8c);
            uint32_t value = REG32(MBOX + 0x90);
            cbo_inval(MBOX + 0x80);
            fence();
            if (REG32(MBOX + 0x80) != seq) {
                continue;  // request still landing; retry next pass
            }

            set_window(x, y, addr);
            uint32_t probe = REG32(TLB_APERTURE + (addr & WINDOW_MASK));
            uint32_t result = value * 3 + 7;
            uint32_t status;
            if (probe == MAGIC) {
                REG32(TLB_APERTURE + ((addr + 16) & WINDOW_MASK)) = result;
                fence();
                status = 1;
            } else {
                status = 2;
            }

            REG32(MBOX + 0x48) = probe;
            REG32(MBOX + 0x4c) = result;
            REG32(MBOX + 0x44) = status;
            fence();
            REG32(MBOX + 0x40) = seq;
            cbo_flush(MBOX + 0x40);
            fence();
            last_seq = seq;
        }
    }
}

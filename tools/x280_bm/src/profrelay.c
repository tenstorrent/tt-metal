/*
 * profrelay.c - X280 relays the DEVICE PROFILER timestamps to the host (close
 * the loop). A normal workload makes all RISCs on all cores write FW + kernel
 * timestamps into each core's L1 profiler buffer (profiler_msg_t, ~10 KB/core:
 * control_vector[32] + 5 x 2048 B per-RISC buffers, guaranteed FW/kernel markers
 * at buffer[risc]+0x10..0x2F). This FW drains that L1 region from every core over
 * the NoC (§12 scatter-read) and relays it to host pinned memory through the PCIe
 * tile (§13 posted write) — replacing tt-metal's profiler readback.
 *
 * Per core: read `bytes_per_core` from the core's profiler L1 and write it to the
 * host region at offset core*bytes_per_core. Reads are 64 B vle64 flits (latency-
 * bound), ILP-grouped so several overlap; writes are posted vse64 (free). The host
 * region is contiguous so the host can parse each core's profiler_msg_t in place.
 * WRITE-ONLY to the PCIe tile (a read through it hangs the hart) — we only read
 * Tensix L1 (safe) and write host (posted).
 *
 * LIM layout (must match host):
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1_addr
 *                          +0x18 bytes_per_core +0x20 num_cores +0x28 ilp
 *                          +0x30 nonce
 *   RESULTS @ 0x08011040 : hart-0 slot: +0x00 cycles +0x08 bytes +0x10 footer_off
 *                          +0x18 done (= DONE_MAGIC, last)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated Tensix)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_BYTES (MBOX_PARAMS + 0x18)
#define P_NUM_CORES (MBOX_PARAMS + 0x20)
#define P_ILP (MBOX_PARAMS + 0x28)
#define P_NONCE (MBOX_PARAMS + 0x30)

#define RES_CYCLES 0x00
#define RES_BYTES 0x08
#define RES_FOOTER 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0x9D0F11A1DEULL
#define FOOTER_MAGIC 0xF007D09F11E12345ULL

#define WRITE_WIN 200u /* TLB window index for the PCIe (host) write target */

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

int main(uint64_t hartid) {
    uint64_t pcie_enc = *(volatile uint64_t*)P_PCIE_ENC;
    uint64_t host_base = *(volatile uint64_t*)P_HOST_BASE;
    uint64_t prof_l1 = *(volatile uint64_t*)P_PROF_L1;
    uint64_t bytes_per_core = *(volatile uint64_t*)P_BYTES;
    uint64_t num_cores = *(volatile uint64_t*)P_NUM_CORES;
    uint64_t ilp = *(volatile uint64_t*)P_ILP;
    uint64_t nonce = *(volatile uint64_t*)P_NONCE;
    (void)nonce;
    if (ilp != 1 && ilp != 4) {
        ilp = 4;
    }

    /* Only hart 0 relays; others park. */
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    volatile uint64_t* rl = (volatile uint64_t*)MBOX_RESULTS;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t pcie_x = pcie_enc & 0x3f;
    uint64_t pcie_y = (pcie_enc >> 6) & 0x3f;
    uint64_t off_r = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t off_w = host_base & (NOC_2M_WINDOW_STRIDE - 1ULL);

    /* Write window -> PCIe tile, host IOVA, POSTED, NoC0. */
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
    (void)noc_configure_tlb_2m_ext(WRITE_WIN, &wt, 0);

    /* Pre-map one read window per core (index = core index) -> Tensix profiler L1. */
    for (uint64_t c = 0; c < num_cores; c++) {
        (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], prof_l1, 0, 0);
    }
    __asm__ volatile("fence iorw, iorw");

    uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)WRITE_WIN * NOC_2M_WINDOW_STRIDE + off_w;
    uint64_t nflits = bytes_per_core / 64;
    nflits &= ~3ULL; /* multiple of 4 */
    if (nflits == 0) {
        nflits = 4;
    }

    uint64_t t0 = rdcycle();
    for (uint64_t c = 0; c < num_cores; c++) {
        uint64_t rbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + off_r;
        uint64_t wcore = wbase + c * bytes_per_core;
        if (ilp == 1) {
            for (uint64_t f = 0; f < nflits; f++) {
                uint64_t rp = rbase + f * 64, wp = wcore + f * 64;
                __asm__ volatile(
                    "vsetivli zero, 8, e64, m1, ta, ma\n"
                    "vle64.v v0, (%0)\n"
                    "vse64.v v0, (%1)\n"
                    :
                    : "r"(rp), "r"(wp)
                    : "memory", "v0");
            }
        } else { /* ilp == 4: overlap 4 reads, then 4 posted writes */
            for (uint64_t f = 0; f + 4 <= nflits; f += 4) {
                uint64_t rp = rbase + f * 64, wp = wcore + f * 64;
                __asm__ volatile(
                    "vsetivli zero, 8, e64, m1, ta, ma\n"
                    "vle64.v v0, (%0)\n"
                    "addi t2, %0, 64\n vle64.v v1, (t2)\n"
                    "addi t2, %0, 128\n vle64.v v2, (t2)\n"
                    "addi t2, %0, 192\n vle64.v v3, (t2)\n"
                    "vse64.v v0, (%1)\n"
                    "addi t2, %1, 64\n vse64.v v1, (t2)\n"
                    "addi t2, %1, 128\n vse64.v v2, (t2)\n"
                    "addi t2, %1, 192\n vse64.v v3, (t2)\n"
                    :
                    : "r"(rp), "r"(wp)
                    : "memory", "t2", "v0", "v1", "v2", "v3");
            }
        }
    }
    uint64_t t1 = rdcycle();

    /* Fence, then footer flit = final posted write (host polls it in sysmem). */
    __asm__ volatile("fence iorw, iorw");
    uint64_t footer_off = num_cores * bytes_per_core;
    uint64_t fptr = wbase + footer_off;
    uint64_t fpat = FOOTER_MAGIC;
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vmv.v.x v1, %0\n"
        "vse64.v v1, (%1)\n"
        :
        : "r"(fpat), "r"(fptr)
        : "memory", "v1");
    __asm__ volatile("fence iorw, iorw");

    *(volatile uint64_t*)(MBOX_RESULTS + RES_CYCLES) = (t1 - t0);
    *(volatile uint64_t*)(MBOX_RESULTS + RES_BYTES) = num_cores * bytes_per_core;
    *(volatile uint64_t*)(MBOX_RESULTS + RES_FOOTER) = footer_off;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)(MBOX_RESULTS + RES_DONE) = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

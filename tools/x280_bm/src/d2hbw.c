/*
 * d2hbw.c - X280 -> HOST device-to-host (D2H) write bandwidth, bare-metal.
 *
 * The export half of the profiler: the X280 fabricates fake 64 B packets and
 * blasts them over the NoC, through the PCIe tile, into host pinned memory
 * (sysmem channel 0). Measures the peak posted-write throughput out to host.
 *
 * Mechanism (proven D2H path, see tools/tracy/x280/X280_BANDWIDTH_SUMMARY.md):
 *   - Target = the PCIe tile's TRANSLATED coord (host passes pcie_enc = x|(y<<6)).
 *   - NoC addr = the host IOVA (get_pcie_base_addr_from_device + offset); a posted
 *     write to (PCIe tile, IOVA) lands in host hugepage memory.
 *   - WRITE-ONLY: a NoC *read* through the PCIe tile hangs the in-order hart, so
 *     this FW issues ZERO reads through the window. Posted writes don't stall, so
 *     one hart can stream stores back-to-back; harts add parallel injection.
 *
 * Each hart h owns window index h + a 2 MiB host region (host_base + h*win_stride,
 * 2 MiB aligned -> window offset 0). It splats a fake packet into v0 once, then
 * stores it `ilp`-at-a-time across its region for `nrounds` passes (timed). After
 * the data, a fence + a single FOOTER flit (v1) is the last posted write; the host
 * polls that footer in sysmem to know all data landed (posted writes to the same
 * PCIe function stay ordered), then verifies + reports BW. No flow control.
 *
 * LIM layout (must match host):
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 win_stride
 *                          +0x18 bytes_per_hart +0x20 nharts +0x28 ilp
 *                          +0x30 nrounds +0x38 nonce (data pattern seed)
 *   RESULTS @ 0x08011040 : per-hart slot h at +h*0x40:
 *                          +0x00 u64 cycles +0x08 u64 bytes +0x10 u64 footer_off
 *                          +0x18 u64 done (= DONE_MAGIC, last)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_WIN_STRIDE (MBOX_PARAMS + 0x10)
#define P_BYTES (MBOX_PARAMS + 0x18)
#define P_NHARTS (MBOX_PARAMS + 0x20)
#define P_ILP (MBOX_PARAMS + 0x28)
#define P_NROUNDS (MBOX_PARAMS + 0x30)
#define P_NONCE (MBOX_PARAMS + 0x38)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_CYCLES 0x00
#define RES_BYTES 0x08
#define RES_FOOTER 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0xD2A011BDDEULL
#define FOOTER_MAGIC 0xF007E2D2D2D2D2D2ULL

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

int main(uint64_t hartid) {
    uint64_t pcie_enc = *(volatile uint64_t*)P_PCIE_ENC;
    uint64_t host_base = *(volatile uint64_t*)P_HOST_BASE;
    uint64_t win_stride = *(volatile uint64_t*)P_WIN_STRIDE;
    uint64_t bytes = *(volatile uint64_t*)P_BYTES;
    uint64_t nharts = *(volatile uint64_t*)P_NHARTS;
    uint64_t ilp = *(volatile uint64_t*)P_ILP;
    uint64_t nrounds = *(volatile uint64_t*)P_NROUNDS;
    uint64_t nonce = *(volatile uint64_t*)P_NONCE;
    if (nharts == 0 || nharts > 4) {
        nharts = 4;
    }
    if (ilp != 1 && ilp != 2 && ilp != 4 && ilp != 8) {
        ilp = 4;
    }
    if (nrounds == 0) {
        nrounds = 1;
    }
    if (bytes < 512) {
        bytes = 512;
    }
    uint64_t slot = RES_SLOT(hartid);

    volatile uint64_t* rl = (volatile uint64_t*)slot;
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    __asm__ volatile("fence iorw, iorw");

    if (hartid >= nharts) {
        __asm__ volatile("fence iorw, iorw");
        *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    uint64_t pcie_x = pcie_enc & 0x3f;
    uint64_t pcie_y = (pcie_enc >> 6) & 0x3f;
    uint64_t host_addr = host_base + hartid * win_stride; /* 2 MiB aligned -> off 0 */

    /* Program window `hartid` -> PCIe tile, host IOVA, POSTED, NoC0. */
    noc_tlb_2m_t tlb;
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.data[3] = 0;
    tlb.addr = host_addr >> 21;
    tlb.x_end = (uint32_t)pcie_x;
    tlb.y_end = (uint32_t)pcie_y;
    tlb.x_start = (uint32_t)pcie_x;
    tlb.y_start = (uint32_t)pcie_y;
    tlb.posted = 1;
    tlb.noc_selector = 0;
    (void)noc_configure_tlb_2m_ext((uint32_t)hartid, &tlb, 0);
    __asm__ volatile("fence iorw, iorw");

    uint64_t off = host_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t base = NOC_2M_WINDOW_BASE + hartid * NOC_2M_WINDOW_STRIDE + off;
    uint64_t footer_off = win_stride - 64; /* last flit of the 2 MiB region */
    uint64_t footer_ptr = NOC_2M_WINDOW_BASE + hartid * NOC_2M_WINDOW_STRIDE + off + footer_off;

    uint64_t nflits = bytes / 64;
    nflits &= ~7ULL; /* multiple of 8 */
    if (nflits == 0) {
        nflits = 8;
    }
    uint64_t dpat = nonce ^ (hartid * 0x0101010101010101ULL);

    /* Splat the fake packet into v0 (8x u64 = one 64 B flit), once. */
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vmv.v.x v0, %0\n"
        :
        : "r"(dpat)
        : "v0");

    uint64_t t0 = rdcycle();
    for (uint64_t round = 0; round < nrounds; round++) {
        uint64_t p = base;
        uint64_t n = nflits;
        if (ilp == 1) {
            __asm__ volatile("1:\n vse64.v v0, (%0)\n addi %0, %0, 64\n addi %1, %1, -1\n bnez %1, 1b\n"
                             : "+r"(p), "+r"(n)
                             :
                             : "memory");
        } else if (ilp == 2) {
            __asm__ volatile(
                "1:\n vse64.v v0, (%0)\n addi t2, %0, 64\n vse64.v v0, (t2)\n"
                "addi %0, %0, 128\n addi %1, %1, -2\n bnez %1, 1b\n"
                : "+r"(p), "+r"(n)
                :
                : "memory", "t2");
        } else if (ilp == 4) {
            __asm__ volatile(
                "1:\n vse64.v v0, (%0)\n addi t2, %0, 64\n vse64.v v0, (t2)\n"
                "addi t3, %0, 128\n vse64.v v0, (t3)\n addi t4, %0, 192\n vse64.v v0, (t4)\n"
                "addi %0, %0, 256\n addi %1, %1, -4\n bnez %1, 1b\n"
                : "+r"(p), "+r"(n)
                :
                : "memory", "t2", "t3", "t4");
        } else { /* ilp == 8 */
            __asm__ volatile(
                "1:\n vse64.v v0, (%0)\n addi t2, %0, 64\n vse64.v v0, (t2)\n"
                "addi t2, %0, 128\n vse64.v v0, (t2)\n addi t2, %0, 192\n vse64.v v0, (t2)\n"
                "addi t2, %0, 256\n vse64.v v0, (t2)\n addi t2, %0, 320\n vse64.v v0, (t2)\n"
                "addi t2, %0, 384\n vse64.v v0, (t2)\n addi t2, %0, 448\n vse64.v v0, (t2)\n"
                "addi %0, %0, 512\n addi %1, %1, -8\n bnez %1, 1b\n"
                : "+r"(p), "+r"(n)
                :
                : "memory", "t2");
        }
    }
    uint64_t t1 = rdcycle();

    /* Order all data writes, then the footer flit = the final posted write. */
    __asm__ volatile("fence iorw, iorw");
    uint64_t fpat = FOOTER_MAGIC;
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vmv.v.x v1, %0\n"
        "vse64.v v1, (%1)\n"
        :
        : "r"(fpat), "r"(footer_ptr)
        : "memory", "v1");
    __asm__ volatile("fence iorw, iorw");

    *(volatile uint64_t*)(slot + RES_CYCLES) = (t1 - t0);
    *(volatile uint64_t*)(slot + RES_BYTES) = nflits * 64ULL * nrounds;
    *(volatile uint64_t*)(slot + RES_FOOTER) = footer_off;
    __asm__ volatile("fence iorw, iorw");
    *(volatile uint64_t*)(slot + RES_DONE) = DONE_MAGIC;
    __asm__ volatile("fence iorw, iorw");

    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

/*
 * profsock.c - X280 D2H-socket SENDER bench.
 *
 * The host (extended D2HSocket with sender_is_l2cpu) allocates a pinned host FIFO and
 * writes the `sender_socket_md` into this X280's LIM at P_CONFIG_ADDR. This FW reads that
 * config and acts as the socket SENDER: it pushes synthetic 64 B pages through the socket
 * to the host FIFO via the PCIe tile, running the real reserve/push/notify protocol
 * (socket_api.h), and times the push so the host can compute sustained socket BW and
 * compare it to the raw-relay ~1.2 GB/s baseline (the "BW hit" of the page/socket model).
 *
 * sender_socket_md word layout @ config_addr (L1_ALIGNMENT=16 => md 32B, ack 16B, enc 16B):
 *   [0] bytes_sent  [1] num_downstreams  [2] write_ptr  [3] dn_bytes_sent_addr_lo
 *   [4] dn_fifo_addr_lo  [5] fifo_total_size  [6] is_d2h
 *   [8] bytes_acked (host writes acks here)
 *   [12] bytes_sent_addr_hi  [13] fifo_addr_hi  [14] pcie_xy_enc
 *
 * Params @ MBOX_PARAMS: +0x00 config_addr  +0x08 pcie_x  +0x10 pcie_y  +0x18 npages
 * Results @ MBOX_RESULTS: +0x00 bytes pushed  +0x08 cycles  +0x18 done(=DONE_MAGIC)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define P_CONFIG_ADDR (MBOX_PARAMS + 0x00)
#define P_PCIE_X (MBOX_PARAMS + 0x08)
#define P_PCIE_Y (MBOX_PARAMS + 0x10)
#define P_NPAGES (MBOX_PARAMS + 0x18)
#define P_BATCH (MBOX_PARAMS + 0x20) /* pages to push per single notify (amortizes reserve+notify) */
#define RES(o) (MBOX_RESULTS + (o))
#define RES_DONE 0x18
#define DONE_MAGIC 0x5005C0FFEEULL
#define WRITE_WIN 200u
#define PAGE 64u

/* socket_socket_md word offsets */
#define C_BYTES_SENT 0
#define C_WRITE_PTR 2
#define C_DN_BSENT_LO 3
#define C_DN_FIFO_LO 4
#define C_FIFO_TOTAL 5
#define C_BACKED 8 /* bytes_acked, byte offset 32 = word 8 */
#define C_BSENT_HI 12
#define C_FIFO_HI 13

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
/* posted 64 B write of v0 to wp (host FIFO via PCIe tile) */
static inline void vwr64(uint64_t wp) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vse64.v v0, (%0)\n" : : "r"(wp) : "memory", "v0");
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    uint64_t cfg = r64(P_CONFIG_ADDR);
    uint64_t pcie_x = r64(P_PCIE_X), pcie_y = r64(P_PCIE_Y);
    uint64_t npages = r64(P_NPAGES);

    volatile uint32_t* c = (volatile uint32_t*)cfg;
    uint32_t write_ptr = c[C_WRITE_PTR];
    uint32_t fifo_total = c[C_FIFO_TOTAL];
    uint64_t fifo_addr = ((uint64_t)c[C_FIFO_HI] << 32) | c[C_DN_FIFO_LO];
    uint64_t bsent_addr = ((uint64_t)c[C_BSENT_HI] << 32) | c[C_DN_BSENT_LO];
    uint64_t backed_addr = cfg + (uint64_t)C_BACKED * 4; /* bytes_acked in our LIM */
    uint32_t bytes_sent = c[C_BYTES_SENT];

    /* NOC1 posted TLB window to the PCIe tile, mapping the host FIFO's 2 MiB window */
    noc_tlb_2m_t wt;
    wt.data[0] = 0;
    wt.data[1] = 0;
    wt.data[2] = 0;
    wt.data[3] = 0;
    wt.addr = fifo_addr >> 21;
    wt.x_end = (uint32_t)pcie_x;
    wt.y_end = (uint32_t)pcie_y;
    wt.x_start = (uint32_t)pcie_x;
    wt.y_start = (uint32_t)pcie_y;
    wt.posted = 1;
    wt.noc_selector = 1;
    (void)noc_configure_tlb_2m_ext(WRITE_WIN, &wt, 0);
    fence_();
    uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)WRITE_WIN * NOC_2M_WINDOW_STRIDE;
    uint64_t fifo_off = fifo_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t bsent_off = bsent_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);

    uint32_t batch = (uint32_t)r64(P_BATCH);
    if (batch == 0) {
        batch = 1;
    }
    if (batch > fifo_total / PAGE) {
        batch = fifo_total / PAGE; /* can't reserve more than the FIFO holds */
    }
    uint64_t t0 = rdcycle();
    uint64_t i = 0;
    while (i < npages) {
        uint32_t b = (uint32_t)((npages - i) < batch ? (npages - i) : batch);
        uint32_t need = b * PAGE;
        /* reserve the whole batch once: wait until the host has acked enough free space */
        for (;;) {
            fence_();  // re-read the host-updated bytes_acked in LIM
            uint32_t acked = r32(backed_addr);
            if (fifo_total - (bytes_sent - acked) >= need) {
                break;
            }
        }
        for (uint32_t k = 0; k < b; k++) {
            vwr64(wbase + fifo_off + write_ptr); /* page write (64 B) to host FIFO */
            write_ptr += PAGE;                   /* advance write_ptr (wrap) + bytes_sent */
            if (write_ptr >= fifo_total) {
                write_ptr -= fifo_total;
            }
            bytes_sent += PAGE;
        }
        w32(wbase + bsent_off, bytes_sent); /* ONE notify per batch (publish bytes_sent) */
        i += b;
    }
    uint64_t t1 = rdcycle();
    w64(RES(0x00), npages * PAGE);
    w64(RES(0x08), t1 - t0);
    fence_();
    w64(RES(RES_DONE), DONE_MAGIC);
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

/*
 * profzone.c - X280: drain the SPSC kernel-profiler rings, PAIR start/end markers
 * per (core,risc) ON-DEVICE, and push each complete zone as a device-zone page
 * through a D2H socket (sender = X280 L2CPU). The host drains the socket via
 * socket.read() and emits per-(core,risc) Tracy zones.
 *
 * Single pass over the (static, post-workload) rings: the host runs a SMALL profiled
 * workload whose markers fit in the 512-word rings (so producers never block without a
 * drainer), then boots this FW to drain+pair+push. Pairing uses a local per-ring stack
 * (zones nest within a ring), so no persistent LIM stacks are needed.
 *
 * SPSC contract (profiler_msg_t @ prof_l1): control_vector[32] then buffer[r] @ +128 +
 * r*2048; head=ctrl[r], tail=ctrl[5+r] (monotonic WORD counts), storage idx = count%512.
 * Marker = 2 words: w0 = 0x80000000 | (timer_id<<12) | time_H, w1 = time_L.
 * packet type = (timer_id>>16)&7 (0=ZONE_START, 1=ZONE_END).
 *
 * Params @ MBOX_PARAMS: +0x00 config_addr +0x08 pcie_x +0x10 pcie_y +0x18 prof_l1
 *                       +0x20 num_cores
 * Results @ MBOX_RESULTS: +0x00 total_zones  +0x18 done(=DONE_MAGIC)
 * Coords @ MBOX_COORDS: num_cores x {u32 noc_x, u32 noc_y} (translated)
 */
#include <stdint.h>

#include "noc.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define P_CONFIG_ADDR (MBOX_PARAMS + 0x00)
#define P_PCIE_X (MBOX_PARAMS + 0x08)
#define P_PCIE_Y (MBOX_PARAMS + 0x10)
#define P_PROF_L1 (MBOX_PARAMS + 0x18)
#define P_NUM_CORES (MBOX_PARAMS + 0x20)
#define RES(o) (MBOX_RESULTS + (o))
#define RES_DONE 0x18
#define DONE_MAGIC 0x20E50FFEE1ULL

#define NRISC 5
#define RING_CAP 512u
#define WRITE_WIN 200u
#define PAGE 64u
#define STACK_DEPTH 32
#define CTRL_HEAD(r) (r)
#define CTRL_TAIL(r) (5u + (r))

/* socket sender_socket_md word offsets @ config_addr */
#define C_BYTES_SENT 0
#define C_WRITE_PTR 2
#define C_DN_BSENT_LO 3
#define C_DN_FIFO_LO 4
#define C_FIFO_TOTAL 5
#define C_BACKED 8
#define C_BSENT_HI 12
#define C_FIFO_HI 13

static inline uint32_t r32(uint64_t a) { return *(volatile uint32_t*)a; }
static inline void w32(uint64_t a, uint32_t v) { *(volatile uint32_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }

int main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    uint64_t cfg = r64(P_CONFIG_ADDR);
    uint64_t pcie_x = r64(P_PCIE_X), pcie_y = r64(P_PCIE_Y);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);

    volatile uint32_t* c = (volatile uint32_t*)cfg;
    uint32_t write_ptr = c[C_WRITE_PTR];
    uint32_t fifo_total = c[C_FIFO_TOTAL];
    uint64_t fifo_addr = ((uint64_t)c[C_FIFO_HI] << 32) | c[C_DN_FIFO_LO];
    uint64_t bsent_addr = ((uint64_t)c[C_BSENT_HI] << 32) | c[C_DN_BSENT_LO];
    uint64_t backed_addr = cfg + (uint64_t)C_BACKED * 4;
    uint32_t bytes_sent = c[C_BYTES_SENT];

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
    uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)WRITE_WIN * NOC_2M_WINDOW_STRIDE;
    uint64_t fifo_off = fifo_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t bsent_off = bsent_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);

    for (uint64_t cc = 0; cc < num_cores; cc++) {
        (void)noc_configure_tlb_2m((uint32_t)cc, coords[cc * 2 + 0], coords[cc * 2 + 1], prof_l1, 0, 0);
    }
    fence_();

    uint64_t total_zones = 0;
    for (uint64_t cc = 0; cc < num_cores; cc++) {
        uint64_t cbase = NOC_2M_WINDOW_BASE + cc * NOC_2M_WINDOW_STRIDE + ctrl_off;
        uint64_t rbufs = cbase + 128;
        uint32_t core_x = coords[cc * 2 + 0], core_y = coords[cc * 2 + 1];
        for (uint32_t r = 0; r < NRISC; r++) {
            uint32_t tail = r32(cbase + CTRL_TAIL(r) * 4);
            uint32_t head = r32(cbase + CTRL_HEAD(r) * 4);
            uint64_t ring_base = rbufs + (uint64_t)r * 2048;
            uint64_t stk_ts[STACK_DEPTH];
            uint32_t stk_id[STACK_DEPTH];
            int sp = 0;
            uint32_t h = head;
            while (h != tail) {
                uint32_t w0 = r32(ring_base + (uint64_t)(h % RING_CAP) * 4);
                uint32_t w1 = r32(ring_base + (uint64_t)((h + 1) % RING_CAP) * 4);
                h += 2;
                if ((w0 & 0x80000000u) == 0) {
                    continue;
                }
                uint32_t timer_id = (w0 >> 12) & 0x7FFFF;
                uint32_t ptype = (timer_id >> 16) & 0x7;
                uint64_t ts = ((uint64_t)(w0 & 0xFFF) << 32) | w1;
                if (ptype == 0) {
                    if (sp < STACK_DEPTH) {
                        stk_ts[sp] = ts;
                        stk_id[sp] = timer_id & 0xFFFF;
                        sp++;
                    }
                } else if (ptype == 1 && sp > 0) {
                    sp--;
                    for (;;) {
                        fence_();
                        uint32_t acked = r32(backed_addr);
                        if (fifo_total - (bytes_sent - acked) >= PAGE) {
                            break;
                        }
                    }
                    uint64_t p = wbase + fifo_off + write_ptr;
                    uint64_t st = stk_ts[sp];
                    w32(p + 0, (uint32_t)(st >> 32));
                    w32(p + 4, (uint32_t)st);
                    w32(p + 8, (uint32_t)(ts >> 32));
                    w32(p + 12, (uint32_t)ts);
                    w32(p + 16, core_x);
                    w32(p + 20, core_y);
                    w32(p + 24, r);
                    w32(p + 28, stk_id[sp]);
                    write_ptr += PAGE;
                    if (write_ptr >= fifo_total) {
                        write_ptr -= fifo_total;
                    }
                    bytes_sent += PAGE;
                    w32(wbase + bsent_off, bytes_sent);
                    total_zones++;
                }
            }
            w32(cbase + CTRL_HEAD(r) * 4, h);
        }
    }
    fence_();
    w64(RES(0x00), total_zones);
    fence_();
    w64(RES(RES_DONE), DONE_MAGIC);
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}

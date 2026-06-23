// X280 FULL-PIPELINE relay: Tensix L1 SPSC ring --pull--> X280 --push--> host
// pinned FIFO (D2H socket, via the PCIe tile). Single hart does both hops (the
// X280 has no DMA, so every flit is load-ed in then store-d out).
//
//   pull side  (window 0, Tensix L1, RW): read w; read flit r; write r back
//              (frees the Tensix producer). Same SPSC protocol as the consumers.
//   push side  (window 1, PCIe tile, RW): write the flit as one page to the host
//              FIFO[write_ptr] (wrap), bump cumulative bytes_sent. Flow control
//              reads bytes_acked from the socket config in Tensix L1 (host writes
//              it on read(notify=true)) so the host FIFO never overruns.
//
// One 64B flit == one D2H page. WRITE-ONLY through the PCIe tile (no PCIe-tile
// reads -> no hart hang); PCIe coord = pcie_xy_enc verbatim, winsel 0, no bit-60.
//
// usage (root): ./x280_pipe <tx> <ty> <config_addr> \
//                 [ring_base=0x80000] [cells=32] [w=0x80800] [r=0x80840] [max_secs=30]
// config_addr / Tensix coord are printed by the host `zonepipe` mode.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define WINDOW_2M_MASK (WINDOW_2M_SIZE - 1)

// D2H socket config word layout (matches x280_d2h.cpp / x280_d2h_stream.c).
#define W_BSENT_LO 3
#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_IS_D2H 6
#define W_BSENT_HI 12
#define W_DATA_HI 13
#define W_PCIE_ENC 14
#define ACK_OFF 32  // bytes_acked at config_addr + 32 (word 8)

#define PAGE 64u  // one 64B NoC flit per D2H page

static volatile uint32_t* g_cfg;

static void program_window(int win, uint64_t noc_addr, uint32_t xy_enc) {
    volatile uint32_t* reg = g_cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(noc_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(noc_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = xy_enc & 0xFFF;
    reg[3] = 0;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tx> <ty> <config_addr> [ring_base] [cells] [w] [r] [max_secs]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    uint64_t ring_base = argc > 4 ? strtoull(argv[4], 0, 0) : 0x80000UL;
    uint32_t cells = argc > 5 ? (uint32_t)strtoul(argv[5], 0, 0) : 32u;
    uint64_t w_addr = argc > 6 ? strtoull(argv[6], 0, 0) : 0x80800UL;
    uint64_t r_addr = argc > 7 ? strtoull(argv[7], 0, 0) : 0x80840UL;
    double max_secs = argc > 8 ? atof(argv[8]) : 30.0;
    const uint32_t mask = cells - 1;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    g_cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);

    // window 0: Tensix (tx,ty) L1 from offset 0 (covers ring + socket config +
    // bytes_acked). RW: we read ring/config and WRITE r back to free the producer.
    program_window(0, 0, (tx & 0x3f) | ((ty & 0x3f) << 6));
    volatile uint8_t* w0 = mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (w0 == MAP_FAILED) {
        perror("mmap w0");
        return 1;
    }
    volatile uint32_t* ring = (volatile uint32_t*)(w0 + (ring_base & WINDOW_2M_MASK));
    volatile uint32_t* wptr = (volatile uint32_t*)(w0 + (w_addr & WINDOW_2M_MASK));
    volatile uint32_t* rptr = (volatile uint32_t*)(w0 + (r_addr & WINDOW_2M_MASK));
    volatile uint32_t* cfg = (volatile uint32_t*)(w0 + (cfg_addr & WINDOW_2M_MASK));
    volatile uint32_t* ack_p = (volatile uint32_t*)(w0 + ((cfg_addr + ACK_OFF) & WINDOW_2M_MASK));

    uint32_t c[16];
    for (int i = 0; i < 16; i++) {
        c[i] = cfg[i];
    }
    uint64_t host_data = ((uint64_t)c[W_DATA_HI] << 32) | c[W_FIFO_LO];
    uint64_t host_bsent = ((uint64_t)c[W_BSENT_HI] << 32) | c[W_BSENT_LO];
    uint32_t pcie_enc = c[W_PCIE_ENC], fifo = c[W_FIFO_SZ], is_d2h = c[W_IS_D2H];
    printf(
        "pipe: Tensix (%u,%u) ring=0x%lx cfg=0x%lx | host_data=0x%lx host_bsent=0x%lx pcie=0x%x fifo=%u d2h=%u\n",
        tx,
        ty,
        ring_base,
        cfg_addr,
        host_data,
        host_bsent,
        pcie_enc,
        fifo,
        is_d2h);
    if (!is_d2h || host_data == 0) {
        fprintf(stderr, "no live D2H socket (run host `zonepipe`)\n");
        return 2;
    }
    if (PAGE > fifo || (fifo % PAGE) || ((host_data + fifo) >> WINDOW_2M_SHIFT) != (host_data >> WINDOW_2M_SHIFT) ||
        (host_bsent >> WINDOW_2M_SHIFT) != (host_data >> WINDOW_2M_SHIFT)) {
        fprintf(stderr, "fifo/page/window geometry bad (fifo=%u page=%u)\n", fifo, PAGE);
        return 2;
    }

    // window 1: host FIFO + bytes_sent through the PCIe tile (write path).
    program_window(1, host_data, pcie_enc);
    volatile uint8_t* w1 =
        mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + WINDOW_2M_SIZE);
    if (w1 == MAP_FAILED) {
        perror("mmap w1");
        return 1;
    }
    volatile uint8_t* fifo_base = w1 + (host_data & WINDOW_2M_MASK);
    volatile uint32_t* bsp = (volatile uint32_t*)(w1 + (host_bsent & WINDOW_2M_MASK));

    uint32_t ring_r = *rptr;  // oldest unread (producer blocks-when-full => [r,w) valid)
    uint32_t write_ptr = 0;
    uint64_t bytes_sent = 0, acked = 0, flits = 0, spins = 0;
    double t0 = now_s(), t_end = t0 + max_secs;

    for (;;) {
        const uint32_t w = *wptr;  // pull: read producer's write index
        if ((int32_t)(w - ring_r) <= 0) {
            if (now_s() > t_end) {
                break;  // ring drained + producer done (or timeout)
            }
            continue;
        }
        // push flow control: keep one page of FIFO headroom vs host bytes_acked
        while (bytes_sent - acked > (uint64_t)(fifo - PAGE)) {
            acked = ack_p[0];  // safe Tensix L1 read of host-written bytes_acked
            if (now_s() > t_end) {
                goto done;
            }
            spins++;
        }
        // pull one flit (8x u64) from the ring into registers
        volatile uint64_t* src = (volatile uint64_t*)(ring + (ring_r & mask) * 16u);
        uint64_t f0 = src[0], f1 = src[1], f2 = src[2], f3 = src[3];
        uint64_t f4 = src[4], f5 = src[5], f6 = src[6], f7 = src[7];
        // push the flit as one page to host FIFO[write_ptr] (posted PCIe writes)
        volatile uint64_t* dp = (volatile uint64_t*)(fifo_base + write_ptr);
        dp[0] = f0;
        dp[1] = f1;
        dp[2] = f2;
        dp[3] = f3;
        dp[4] = f4;
        dp[5] = f5;
        dp[6] = f6;
        dp[7] = f7;
        __sync_synchronize();
        write_ptr += PAGE;
        if (write_ptr >= fifo) {
            write_ptr = 0;
        }
        bytes_sent += PAGE;
        *bsp = (uint32_t)bytes_sent;  // notify host (cumulative bytes_sent)
        __sync_synchronize();
        ring_r++;
        *rptr = ring_r;  // free the Tensix producer
        flits++;
    }
done:;
    double dt = now_s() - t0;
    printf(
        "PIPE relayed %lu flits (%lu B) in %.3fs -> %.1f MB/s (spins=%lu, last acked=%lu, w=%u r=%u)\n",
        flits,
        bytes_sent,
        dt,
        bytes_sent / 1e6 / dt,
        spins,
        acked,
        *wptr,
        ring_r);
    return 0;
}

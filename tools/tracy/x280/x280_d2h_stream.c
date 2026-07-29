// X280 -> HOST flow-controlled streaming over a D2H socket (bh-14/bh-03 path).
//
// The lossless, wrapping version of x280_d2h_send.c: stream many pages through a
// ring FIFO with real flow control, so the host drains every page (no overwrite).
//
// Flow-control protocol (matches the Tensix pcie_socket_sender.cpp + D2HSocket):
//   - The host `serve` loop calls D2HSocket::read(..., notify_sender=true) which,
//     after consuming each page, WRITES the cumulative bytes_acked back to the
//     sender core's L1 config buffer at config_addr + ACK_OFF (word 8 / +32 B).
//   - We READ that bytes_acked from Tensix L1 over NoC (a SAFE Tensix read) to
//     know how much FIFO space is free, and never let bytes_sent run more than
//     (fifo_size - page) ahead of bytes_acked -> the host never loses a page.
//   - We WRITE each page to host FIFO[write_ptr] through the PCIe tile (posted),
//     advance write_ptr with wrap, then WRITE the cumulative bytes_sent to the
//     host bytes_sent word so the host's has_data()/read() releases the page.
//
// WRITE-ONLY through the PCIe tile (no PCIe-tile reads -> no hart hang on bh-14).
// PCIe coord = pcie_xy_enc verbatim, winsel 0, no bit-60 (the proven addressing).
//
// usage (root): ./x280_d2h_stream <tx> <ty> <config_addr> [total_MB=256] [page=4096] [max_secs=20]
//   The host must run `serve <fifo> <page> <secs>` with the SAME page and a fifo
//   that fits one 2MB window from host_data (e.g. 1 MB).

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

#define W_BSENT_LO 3
#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_IS_D2H 6
#define W_BSENT_HI 12
#define W_DATA_HI 13
#define W_PCIE_ENC 14
#define ACK_OFF 32  // bytes_acked lives at config_addr + sender md_size (word 8)

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
        fprintf(stderr, "usage: %s <tx> <ty> <config_addr> [total_MB] [page] [max_secs]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    uint64_t total = (argc > 4 ? strtoull(argv[4], 0, 0) : 256) * 1024 * 1024;
    uint32_t page = argc > 5 ? (uint32_t)strtoul(argv[5], 0, 0) : 4096;
    double max_secs = argc > 6 ? atof(argv[6]) : 20.0;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    g_cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);

    // window 0: Tensix sender L1 (config + bytes_acked). Plain Tensix coord, safe reads.
    program_window(0, cfg_addr, (tx & 0x3f) | ((ty & 0x3f) << 6));
    volatile uint8_t* w0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
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
        "host_data=0x%lx host_bsent=0x%lx pcie_enc=0x%x fifo=%u is_d2h=%u | total=%luMB page=%u\n",
        host_data,
        host_bsent,
        pcie_enc,
        fifo,
        is_d2h,
        total / 1024 / 1024,
        page);
    if (!is_d2h || host_data == 0) {
        fprintf(stderr, "no live D2H socket (run host `serve`)\n");
        return 2;
    }
    if (page > fifo || (fifo % page)) {
        fprintf(stderr, "page %u must divide fifo %u\n", page, fifo);
        return 2;
    }
    // FIFO + bytes_sent word must sit in one 2MB window from host_data.
    if (((host_data + fifo) >> WINDOW_2M_SHIFT) != (host_data >> WINDOW_2M_SHIFT) ||
        ((host_bsent >> WINDOW_2M_SHIFT) != (host_data >> WINDOW_2M_SHIFT))) {
        fprintf(stderr, "fifo/bytes_sent span >1 2MB window from base 0x%lx; shrink fifo\n", host_data);
        return 2;
    }

    // window 1: host FIFO + bytes_sent through the PCIe tile (write path).
    program_window(1, host_data, pcie_enc);
    volatile uint8_t* w1 =
        mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + WINDOW_2M_SIZE);
    volatile uint8_t* fifo_base = w1 + (host_data & WINDOW_2M_MASK);
    volatile uint32_t* bsp = (volatile uint32_t*)(w1 + (host_bsent & WINDOW_2M_MASK));

    uint32_t pw = page / 8;  // u64 stores per page
    uint32_t write_ptr = 0;
    uint64_t bytes_sent = 0;
    uint64_t acked = 0;
    uint64_t spins = 0;
    double t0 = now_s(), t_end = t0 + max_secs;

    while (bytes_sent < total) {
        // reserve: ensure room for one page (bytes_sent - acked <= fifo - page)
        while (bytes_sent - acked > (uint64_t)(fifo - page)) {
            acked = ack_p[0];  // safe Tensix L1 read of host-written bytes_acked
            if (now_s() > t_end) {
                goto done;
            }
            spins++;
        }
        // write one page of u64 stores to FIFO[write_ptr] (posted PCIe writes)
        volatile uint64_t* dp = (volatile uint64_t*)(fifo_base + write_ptr);
        uint64_t pat = 0xD2D2000000000000ULL | (bytes_sent / page);
        for (uint32_t i = 0; i < pw; i++) {
            dp[i] = pat + i;
        }
        __sync_synchronize();
        write_ptr += page;
        if (write_ptr >= fifo) {
            write_ptr = 0;
        }
        bytes_sent += page;
        *bsp = (uint32_t)bytes_sent;  // notify host (cumulative bytes_sent, 32-bit wrap ok)
        __sync_synchronize();
        if ((bytes_sent & 0xFFFFF) == 0 && now_s() > t_end) {
            break;
        }
    }
done:;
    double dt = now_s() - t0;
    printf(
        "STREAM sent %lu B in %.3fs -> %.1f MB/s (page=%u fifo=%u spins=%lu, last acked=%lu)\n",
        bytes_sent,
        dt,
        bytes_sent / 1e6 / dt,
        page,
        fifo,
        spins,
        acked);
    return 0;
}

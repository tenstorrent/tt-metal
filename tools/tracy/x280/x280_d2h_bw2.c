// X280 -> HOST raw write-bandwidth, multi-hart (bh-14 write-only path).
//
// Measures the X280's posted-NoC-write throughput INTO host memory through the
// PCIe tile -- the fast D2H export ceiling. No flow control, no host read: run
// the host in `hold` mode (it just pins the FIFO and waits), then each X280 hart
// blasts u64 stores at its own disjoint slice of the host FIFO for `secs`.
//
// Addressing is the PROVEN bh-14 path (x280_d2h_send.c): PCIe tile coord =
// pcie_xy_enc (translated, reg[2] verbatim), winsel 0, NO bit-60. WRITE-ONLY:
// X280 NoC reads through the PCIe tile hang the in-order hart on bh-14, so we
// never read host memory.
//
// usage (root): ./x280_d2h_bw2 <tensix_x> <tensix_y> <config_addr> [nharts=1] [secs=5] [chunk=65536]
//   each hart writes `chunk` bytes repeatedly to host_data + hart*chunk.
//   require: nharts*chunk <= fifo_total_size AND fits one 2MB window from host_data.

#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define WINDOW_2M_MASK (WINDOW_2M_SIZE - 1)

#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_IS_D2H 6
#define W_DATA_HI 13
#define W_PCIE_ENC 14

static volatile uint32_t* g_cfg;

// props_lo (reg[2]): x_end[5:0]|y_end[11:6] = xy_enc; noc_sel[31] picks NoC0/NoC1.
#define PROPS_NOC_SEL (1u << 31)
static void program_window(int win, uint64_t noc_addr, uint32_t xy_enc, int noc1) {
    volatile uint32_t* reg = g_cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(noc_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(noc_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = (xy_enc & 0xFFF) | (noc1 ? PROPS_NOC_SEL : 0u);
    reg[3] = 0;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tensix_x> <tensix_y> <config_addr> [nharts] [secs] [chunk]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    int nharts = argc > 4 ? atoi(argv[4]) : 1;
    double secs = argc > 5 ? atof(argv[5]) : 5.0;
    uint32_t chunk = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 0) : 65536;
    int noc_split = argc > 7 ? atoi(argv[7]) : 0;  // 1 = even harts NoC0, odd NoC1

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    g_cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);

    // read socket config from Tensix L1 (safe read)
    program_window(0, cfg_addr, (tx & 0x3f) | ((ty & 0x3f) << 6), 0);
    volatile uint8_t* w0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    volatile uint32_t* c = (volatile uint32_t*)(w0 + (cfg_addr & WINDOW_2M_MASK));
    uint64_t host_data = ((uint64_t)c[W_DATA_HI] << 32) | c[W_FIFO_LO];
    uint32_t fifo_sz = c[W_FIFO_SZ], pcie_enc = c[W_PCIE_ENC], is_d2h = c[W_IS_D2H];
    munmap((void*)w0, WINDOW_2M_SIZE);

    printf(
        "host_data=0x%lx fifo=%u pcie_enc=0x%x is_d2h=%u | nharts=%d secs=%.1f chunk=%u\n",
        host_data,
        fifo_sz,
        pcie_enc,
        is_d2h,
        nharts,
        secs,
        chunk);
    if (!is_d2h || host_data == 0) {
        fprintf(stderr, "no live D2H socket (run host `hold`)\n");
        return 2;
    }
    if ((uint64_t)nharts * chunk > fifo_sz) {
        fprintf(stderr, "nharts*chunk %lu > fifo %u\n", (uint64_t)nharts * chunk, fifo_sz);
        return 2;
    }
    if (((host_data + (uint64_t)nharts * chunk - 1) >> WINDOW_2M_SHIFT) != (host_data >> WINDOW_2M_SHIFT)) {
        fprintf(stderr, "slices span >1 2MB window from base 0x%lx; reduce nharts*chunk\n", host_data);
        return 2;
    }

    // shared result array (bytes written per hart)
    volatile uint64_t* res = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    double t_launch = now_s();
    for (int h = 0; h < nharts; h++) {
        pid_t pid = fork();
        if (pid == 0) {
            // child = one worker hart
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(h % 4, &set);
            sched_setaffinity(0, sizeof set, &set);

            uint64_t slice_addr = host_data + (uint64_t)h * chunk;
            int win = h + 1;
            int noc1 = noc_split ? (h & 1) : 0;
            program_window(win, slice_addr, pcie_enc, noc1);
            volatile uint8_t* w = mmap(
                0,
                WINDOW_2M_SIZE,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                WINDOW_2M_BASE + (size_t)win * WINDOW_2M_SIZE);
            volatile uint64_t* dp = (volatile uint64_t*)(w + (slice_addr & WINDOW_2M_MASK));
            uint32_t nw = chunk / 8;
            uint64_t pat = 0xD2D2000000000000ULL | ((uint64_t)h << 32);

            uint64_t iters = 0;
            double t0 = now_s(), t1 = t0;
            while (t1 - t0 < secs) {
                for (uint32_t rep = 0; rep < 256; rep++) {  // batch to amortize clock reads
                    for (uint32_t i = 0; i < nw; i++) {
                        dp[i] = pat + i;
                    }
                    iters++;
                }
                t1 = now_s();
            }
            res[h] = iters * (uint64_t)chunk;
            double bw = res[h] / 1e6 / (t1 - t0);
            printf("  hart %d (cpu %d): %.1f MB/s (%lu B in %.2fs)\n", h, h % 4, bw, res[h], t1 - t0);
            fflush(stdout);
            _exit(0);
        }
    }
    for (int h = 0; h < nharts; h++) {
        wait(NULL);
    }
    double wall = now_s() - t_launch;

    uint64_t total = 0;
    for (int h = 0; h < nharts; h++) {
        total += res[h];
    }
    printf("AGGREGATE %d hart(s): %.1f MB/s (%lu B, wall %.2fs)\n", nharts, total / 1e6 / secs, total, wall);
    return 0;
}

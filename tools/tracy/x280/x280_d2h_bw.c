// Raw X280 -> HOST write bandwidth through the PCIe NoC tile.
//
// Measures the ceiling of the fast export path: how fast the X280 can push bytes
// into host pinned memory via the chip's PCIe tile (the D2HSocket data FIFO),
// with NO flow control and NO host read -- just sustained writes. This is the
// PCIe-NoC analogue of netsrc/netsink (which measured the SLIRP TCP path at
// 13.9 MB/s). Single hart, then N harts each writing a disjoint FIFO slice
// through its own TLB window, to find the aggregate port ceiling.
//
// Host side keeps the pinned FIFO alive without reading:
//   env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_SKIP_DRAM_TLBS=1 \
//     ./build_Release/programming_examples/metal_example_x280_d2h 3 hold <fifo> 4096 <secs>
//   (e.g. fifo = 0x100000 = 1 MiB so 3 harts get ~340 KiB slices)
//
// usage (run as root on the X280):
//   ./x280_d2h_bw <tensix_x> <tensix_y> <config_l1_addr> [secs=3] [nharts=1] [pcie_x=2] [pcie_y=0]
//
// Each hart h pins to CPU h, programs window index h to (host_data + h*slice),
// and stores 8-u64-per-flit in a tight loop for <secs>. We report per-hart and
// aggregate MB/s. The FIFO slice each hart touches stays within one 2MB window.

#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define WINDOW_2M_MASK (WINDOW_2M_SIZE - 1)

// config-buffer word offsets (see tt_metal/hw/inc/hostdev/socket.h)
#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_DATA_HI 13

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

// shared result array (one double per hart), written by children, summed by parent
static volatile double* g_mbps;

static void hart_write_bw(int h, uint64_t base_addr, unsigned px, unsigned py, uint32_t slice_bytes, double secs) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(h, &set);
    sched_setaffinity(0, sizeof set, &set);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        _exit(1);
    }
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    // window index h -> (px,py) at base_addr (the hart's disjoint slice)
    volatile uint32_t* reg = cfg + (h * 0x10) / 4;
    reg[0] = (uint32_t)(base_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(base_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = (px & 0x3f) | ((py & 0x3f) << 6);
    reg[3] = 0;
    volatile uint8_t* win =
        mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + (size_t)h * WINDOW_2M_SIZE);
    if (win == MAP_FAILED) {
        perror("mmap win");
        _exit(1);
    }
    volatile uint64_t* p = (volatile uint64_t*)(win + (base_addr & WINDOW_2M_MASK));
    int n64 = slice_bytes / 8;

    // Sustained write loop: 8 independent u64 stores per 64B flit. Re-writes the
    // same slice repeatedly (no flow control) to isolate pure write throughput.
    unsigned long iters = 0;
    double t0 = now_ns(), tend = t0 + secs * 1e9;
    do {
        volatile uint64_t* d = p;
        for (int k = 0; k < n64; k += 8, d += 8) {
            d[0] = 0;
            d[1] = 1;
            d[2] = 2;
            d[3] = 3;
            d[4] = 4;
            d[5] = 5;
            d[6] = 6;
            d[7] = 7;
        }
        iters++;
    } while (now_ns() < tend);
    double ns = now_ns() - t0;
    double bytes = (double)iters * slice_bytes;
    g_mbps[h] = bytes * 1e3 / ns;
    printf(
        "  hart %d: %.1f MB/s (%.1f ns/flit, %lu iters x %u B)\n",
        h,
        bytes * 1e3 / ns,
        ns / (bytes / 64),
        iters,
        slice_bytes);
    fflush(stdout);
    _exit(0);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(
            stderr,
            "usage: %s <tensix_x> <tensix_y> <config_l1_addr> [secs=3] [nharts=1] [pcie_x] [pcie_y]\n",
            argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    double secs = argc > 4 ? atof(argv[4]) : 3.0;
    int nharts = argc > 5 ? atoi(argv[5]) : 1;
    unsigned px = argc > 6 ? atoi(argv[6]) : 2;
    unsigned py = argc > 7 ? atoi(argv[7]) : 0;
    if (nharts < 1) {
        nharts = 1;
    }

    // read the socket config from the Tensix sender L1 to get the host FIFO addr
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    volatile uint32_t* reg = cfg;  // window 0 for the config read
    reg[0] = (uint32_t)(cfg_addr >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (tx & 0x3f) | ((ty & 0x3f) << 6);
    reg[3] = 0;
    volatile uint8_t* win0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    volatile uint32_t* c = (volatile uint32_t*)(win0 + (cfg_addr & WINDOW_2M_MASK));
    uint64_t host_data = ((uint64_t)c[W_DATA_HI] << 32) | c[W_FIFO_LO];
    uint32_t fifo_sz = c[W_FIFO_SZ];
    munmap((void*)win0, WINDOW_2M_SIZE);
    close(fd);

    uint32_t slice = fifo_sz / nharts;
    slice &= ~63u;  // flit-align
    printf(
        "X280->host PCIe write BW: host_data=0x%lx fifo=%u, %d hart(s), slice=%u B, %.1fs, PCIe (%u,%u)\n",
        host_data,
        fifo_sz,
        nharts,
        slice,
        secs,
        px,
        py);
    if (slice < 64) {
        fprintf(stderr, "fifo too small for %d harts\n", nharts);
        return 2;
    }
    // each hart's slice must stay within one 2MB window
    if ((host_data & WINDOW_2M_MASK) + (uint64_t)(nharts - 1) * slice + slice > WINDOW_2M_SIZE) {
        fprintf(
            stderr,
            "warning: slices may cross a 2MB window boundary (host_data offset 0x%lx); "
            "use a smaller fifo or re-run (IOVA varies)\n",
            host_data & WINDOW_2M_MASK);
    }

    g_mbps = mmap(0, sizeof(double) * nharts, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    for (int h = 0; h < nharts; h++) {
        g_mbps[h] = 0;
    }

    pid_t pids[64];
    for (int h = 0; h < nharts; h++) {
        pid_t pid = fork();
        if (pid == 0) {
            hart_write_bw(h, host_data + (uint64_t)h * slice, px, py, slice, secs);
        }
        pids[h] = pid;
    }
    for (int h = 0; h < nharts; h++) {
        waitpid(pids[h], 0, 0);
    }
    double agg = 0;
    for (int h = 0; h < nharts; h++) {
        agg += g_mbps[h];
    }
    printf("AGGREGATE: %.1f MB/s across %d hart(s)\n", agg, nharts);
    return 0;
}

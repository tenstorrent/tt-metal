// Poll the reserved 64B flit on EVERY Tensix core from the X280, as fast as
// possible. Each core runs the counter kernel (16x u32 @ L1 0x80000 incrementing).
// We map one 2MB NoC TLB window per core (X280 has 224), then sweep all cores
// reading their 64B each iteration.
//
// usage: ./pollall <l1_addr> <iters> <coordfile>
//   coordfile: lines "CORE <x> <y>" (the host launcher prints these)
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define MAX_CORES 224
#define FLIT_U32 16  // 64 bytes

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <l1_addr> <iters> <coordfile>\n", argv[0]);
        return 1;
    }
    uint64_t l1_addr = strtoull(argv[1], 0, 0);
    long iters = atol(argv[2]);
    FILE* cf = fopen(argv[3], "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    unsigned cx[MAX_CORES], cy[MAX_CORES];
    int ncores = 0;
    char line[128];
    while (fgets(line, sizeof line, cf) && ncores < MAX_CORES) {
        unsigned x, y;
        if (sscanf(line, "CORE %u %u", &x, &y) == 2) {
            cx[ncores] = x;
            cy[ncores] = y;
            ncores++;
        }
    }
    fclose(cf);
    if (ncores == 0) {
        fprintf(stderr, "no cores parsed\n");
        return 1;
    }

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    // Config registers for windows 0..ncores-1 live within one page (16B each).
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    if (cfg == MAP_FAILED) {
        perror("mmap cfg");
        return 1;
    }
    // One big mapping covering all per-core windows.
    volatile uint8_t* wins = mmap(0, (size_t)ncores * WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (wins == MAP_FAILED) {
        perror("mmap wins");
        return 1;
    }

    // Program window i -> core (cx[i],cy[i]) at l1_addr, and cache each flit ptr.
    volatile uint32_t* flit[MAX_CORES];
    for (int i = 0; i < ncores; i++) {
        volatile uint32_t* reg = cfg + (i * 0x10) / 4;
        reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
        reg[1] = 0;
        reg[2] = (cx[i] & 0x3f) | ((cy[i] & 0x3f) << 6);
        reg[3] = 0;
        flit[i] = (volatile uint32_t*)(wins + (size_t)i * WINDOW_2M_SIZE + (l1_addr & (WINDOW_2M_SIZE - 1)));
    }

    // Sweep all cores, reading the full 64B flit from each.
    volatile uint32_t sink = 0;
    uint32_t lo = 0xffffffff, hi = 0;
    double t0 = now_ns();
    for (long it = 0; it < iters; it++) {
        for (int i = 0; i < ncores; i++) {
            // Read the 64B flit as 8 independent u64 loads in one expression so
            // they stay outstanding together (1 flit transaction) instead of
            // serializing into 16 round trips.
            volatile uint64_t* f = (volatile uint64_t*)flit[i];
            sink ^= (uint32_t)(f[0] ^ f[1] ^ f[2] ^ f[3] ^ f[4] ^ f[5] ^ f[6] ^ f[7]);
        }
    }
    double ns = now_ns() - t0;

    // One more sweep to report observed seqno spread across cores.
    for (int i = 0; i < ncores; i++) {
        uint32_t v = flit[i][0];
        if (v < lo) {
            lo = v;
        }
        if (v > hi) {
            hi = v;
        }
    }

    long flits = (long)iters * ncores;
    printf("polled %d cores x %ld sweeps = %ld flit reads (64B each)\n", ncores, iters, flits);
    printf(
        "total %.3f ms | %.1f ns/flit | %.2f us per full %d-core sweep | %.2f Mflit/s\n",
        ns / 1e6,
        ns / flits,
        ns / iters / 1e3,
        ncores,
        flits * 1e3 / ns);
    printf("seqno across cores: min %u max %u (spread %u); sink=%u\n", lo, hi, hi - lo, sink);
    return 0;
}

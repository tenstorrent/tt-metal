// Like pollall, but reads <nbytes> per core (default 4096 = full L1 buffer)
// instead of a single 64B flit. Sweeps every core in the coordfile, reading each
// core's nbytes region in 64B-flit chunks, each flit as 8 independent u64 loads
// (the overlap-within-flit pattern). Across flits the in-order X280 still
// serializes (one outstanding), so per-core cost ~= (nbytes/64) * per-flit RTT.
//
// Run 3 of these (one taskset per hart over a disjoint core slice) for the
// 3-hart aggregate, same as the 64B sweep.
//
// usage: ./pollalln <l1_addr> <iters> <coordfile> [nbytes=4096]
//   coordfile: lines "CORE <x> <y>"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port, NoC0
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define MAX_CORES 224

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <l1_addr> <iters> <coordfile> [nbytes=4096]\n", argv[0]);
        return 1;
    }
    uint64_t l1_addr = strtoull(argv[1], 0, 0);
    long iters = atol(argv[2]);
    FILE* cf = fopen(argv[3], "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    uint32_t nbytes = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 4096;
    int flits_per_core = nbytes / 64;  // 64B per flit

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
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    if (cfg == MAP_FAILED) {
        perror("mmap cfg");
        return 1;
    }
    volatile uint8_t* wins = mmap(0, (size_t)ncores * WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (wins == MAP_FAILED) {
        perror("mmap wins");
        return 1;
    }

    // Program window i -> core (cx[i],cy[i]); base ptr at l1_addr within window.
    volatile uint64_t* base[MAX_CORES];
    for (int i = 0; i < ncores; i++) {
        volatile uint32_t* reg = cfg + (i * 0x10) / 4;
        reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
        reg[1] = 0;
        reg[2] = (cx[i] & 0x3f) | ((cy[i] & 0x3f) << 6);
        reg[3] = 0;
        base[i] = (volatile uint64_t*)(wins + (size_t)i * WINDOW_2M_SIZE + (l1_addr & (WINDOW_2M_SIZE - 1)));
    }

    volatile uint32_t sink = 0;
    double t0 = now_ns();
    for (long it = 0; it < iters; it++) {
        for (int i = 0; i < ncores; i++) {
            volatile uint64_t* p = base[i];
            for (int k = 0; k < flits_per_core; k++, p += 8) {
                // one 64B flit as 8 independent u64 (overlap within the flit)
                sink ^= (uint32_t)(p[0] ^ p[1] ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7]);
            }
        }
    }
    double ns = now_ns() - t0;

    long flits = (long)iters * ncores * flits_per_core;
    double bytes = (double)flits * 64.0;
    printf("%d cores x %ld iters x %uB = %.1f MB in %.1f ms\n", ncores, iters, nbytes, bytes / 1e6, ns / 1e6);
    printf(
        "  %.1f ns/flit | %.2f MB/s | %.2f us per %d-core x %uB snapshot | sink=%u\n",
        ns / flits,
        bytes * 1e3 / ns,
        ns / iters / 1e3,
        ncores,
        nbytes,
        sink);
    return 0;
}

// Validate that the X280 can ISSUE NoC WRITES via a TLB window (every prior test
// was reads). Writes a known pattern to an UNUSED Tensix L1 scratch address
// through a 2MB uncached System Port window, reads it back through the same
// window, and verifies — then measures write/read bandwidth. This proves the
// write primitive the host-export push path depends on, at low risk (target is
// scratch L1, not the PCIe tile).
//
// Run as root: ./nocwrite <x> <y> <l1_scratch_addr> <nbytes> [iters]
//   pick an L1 addr NOT used by the running kernel (counter uses 0x80000).
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

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <x> <y> <l1_scratch_addr> <nbytes> [iters]\n", argv[0]);
        return 1;
    }
    unsigned x = atoi(argv[1]), y = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    uint32_t nbytes = (uint32_t)strtoul(argv[4], 0, 0);
    long iters = argc > 5 ? atol(argv[5]) : 100000;
    int win = 0;
    int n64 = nbytes / 8;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    volatile uint32_t* reg = cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    reg[3] = 0;
    // Window mapped READ+WRITE so stores become NoC writes.
    volatile uint64_t* w = mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (w == MAP_FAILED) {
        perror("mmap win");
        return 1;
    }
    volatile uint64_t* p = w + (l1_addr & (WINDOW_2M_SIZE - 1)) / 8;

    // --- correctness: write a known pattern, read it back ---
    for (int i = 0; i < n64; i++) {
        p[i] = 0xC0FFEE0000000000ULL | (uint64_t)i;
    }
    int mismatch = 0;
    uint64_t first_bad = 0;
    for (int i = 0; i < n64; i++) {
        uint64_t got = p[i], want = 0xC0FFEE0000000000ULL | (uint64_t)i;
        if (got != want) {
            if (!mismatch) {
                first_bad = got;
            }
            mismatch++;
        }
    }
    printf(
        "write+readback of %u B to (%u,%u):0x%lx via NoC window: %s",
        nbytes,
        x,
        y,
        l1_addr,
        mismatch ? "MISMATCH" : "OK (verified)");
    if (mismatch) {
        printf(" (%d/%d words bad, first got 0x%lx)", mismatch, n64, first_bad);
    }
    printf("\n");
    if (mismatch) {
        return 2;  // don't benchmark if the basic write didn't land
    }

    // --- write bandwidth: 8 u64 per flit ---
    double t0 = now_ns();
    for (long it = 0; it < iters; it++) {
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
    }
    double wns = now_ns() - t0;
    double wbytes = (double)iters * nbytes;
    printf("WRITE: %.1f ns/flit | %.2f MB/s (%ld x %u B)\n", wns / (wbytes / 64), wbytes * 1e3 / wns, iters, nbytes);

    // --- read bandwidth for comparison (same window) ---
    volatile uint64_t sink = 0;
    t0 = now_ns();
    for (long it = 0; it < iters; it++) {
        volatile uint64_t* s = p;
        for (int k = 0; k < n64; k += 8, s += 8) {
            sink ^= s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
        }
    }
    double rns = now_ns() - t0;
    printf(
        "READ : %.1f ns/flit | %.2f MB/s (sink=%lu)\n", rns / (wbytes / 64), wbytes * 1e3 / rns, (unsigned long)sink);
    return 0;
}

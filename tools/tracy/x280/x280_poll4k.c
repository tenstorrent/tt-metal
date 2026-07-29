// Poll an entire 4096-byte Tensix L1 buffer from the X280 via a 2MB NoC TLB
// window, as fast as possible. The Tensix increments u32[0] and u32[1023];
// everything else must stay zero. Run as root: ./poll4k <x> <y> <l1_addr> <iters>
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
#define BUF_BYTES 4096
#define BUF_WORDS64 (BUF_BYTES / 8)

int main(int argc, char** argv) {
    unsigned x = atoi(argv[1]), y = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    long iters = argc > 4 ? atol(argv[4]) : 1000;
    int win = 200;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    volatile uint32_t* cfg =
        mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE + win * 0x10 & ~0xfffUL);
    if (cfg == MAP_FAILED) {
        perror("mmap cfg");
        return 1;
    }
    volatile uint32_t* reg = cfg + ((win * 0x10) & 0xfff) / 4;
    reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    reg[3] = 0;

    volatile uint64_t* w =
        mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE + (uint64_t)win * WINDOW_2M_SIZE);
    if (w == MAP_FAILED) {
        perror("mmap win");
        return 1;
    }
    volatile uint64_t* buf = w + (l1_addr & (WINDOW_2M_SIZE - 1)) / 8;

    uint64_t local[BUF_WORDS64];
    uint32_t first0 = (uint32_t)buf[0];
    long nonzero_middle = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < iters; i++) {
        for (int j = 0; j < BUF_WORDS64; j++) {
            local[j] = buf[j];
        }
        for (int j = 1; j < BUF_WORDS64 - 1; j++) {
            nonzero_middle += (local[j] != 0) | (j == 0);
        }
        nonzero_middle += (local[0] >> 32) != 0;                  // u32[1] must be zero
        nonzero_middle += (uint32_t)local[BUF_WORDS64 - 1] != 0;  // u32[1022] must be zero
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    uint32_t head = (uint32_t)local[0], tail = (uint32_t)(local[BUF_WORDS64 - 1] >> 32);
    printf(
        "%ld polls of %d bytes in %.3f ms: %.1f us/poll, %.2f MB/s\n",
        iters,
        BUF_BYTES,
        ns / 1e6,
        ns / iters / 1e3,
        iters * (double)BUF_BYTES * 1e3 / ns);
    printf(
        "head %u -> %u (+%u), tail %u, head-tail skew %d, middle nonzero %ld\n",
        first0,
        head,
        head - first0,
        tail,
        (int)(head - tail),
        nonzero_middle);
    return 0;
}

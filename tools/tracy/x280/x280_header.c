// Header-vs-payload read experiment on the X280.
//
// Motivates a "poll a small header, then read the full payload" design: measure
// the latency of reading 4 bytes (1x u32), 16 bytes (4x u32), and the full 4096
// byte payload through a 2MB NoC TLB window. Reuses the counter kernel, where
// buf[0] is a fast-incrementing seqno (the "header"); buf[1023] also increments.
//
// Run as root: ./header <x> <y> <l1_addr> <iters>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define PAYLOAD_BYTES 4096

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char** argv) {
    unsigned x = atoi(argv[1]), y = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    long iters = argc > 4 ? atol(argv[4]) : 100000;
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

    volatile uint32_t* w =
        mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE + (uint64_t)win * WINDOW_2M_SIZE);
    if (w == MAP_FAILED) {
        perror("mmap win");
        return 1;
    }
    volatile uint32_t* hdr = w + (l1_addr & (WINDOW_2M_SIZE - 1)) / 4;

    volatile uint32_t sink = 0;  // defeat dead-code elimination of the loads

    // 4 bytes: one u32 read (the header / seqno).
    double t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        sink ^= hdr[0];
    }
    double t_4 = (now_ns() - t0) / iters;

    // 16 bytes: four u32 reads (header + 3 metadata words).
    t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        sink ^= hdr[0] ^ hdr[1] ^ hdr[2] ^ hdr[3];
    }
    double t_16 = (now_ns() - t0) / iters;

    // 64 bytes: one NoC flit, read as 8 independent u64 loads folded together so
    // the compiler can keep them all outstanding (like the 16B case).
    volatile uint64_t* h64 = (volatile uint64_t*)hdr;
    t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        sink ^= (uint32_t)(h64[0] ^ h64[1] ^ h64[2] ^ h64[3] ^ h64[4] ^ h64[5] ^ h64[6] ^ h64[7]);
    }
    double t_64 = (now_ns() - t0) / iters;

    // Full payload: read every one of the 4096 bytes (as u64) and fold into sink
    // so the optimizer can't elide the loads.
    volatile uint64_t* hdr64 = (volatile uint64_t*)hdr;
    long pay_iters = iters / 100 > 0 ? iters / 100 : 1;
    t0 = now_ns();
    for (long i = 0; i < pay_iters; i++) {
        for (int j = 0; j < PAYLOAD_BYTES / 8; j++) {
            sink ^= (uint32_t)hdr64[j];
        }
    }
    double t_pay = (now_ns() - t0) / pay_iters;

    printf("read latency through 2MB NoC window (target %u,%u  addr 0x%lx):\n", x, y, l1_addr);
    printf("   4 B (1x u32) : %7.1f ns\n", t_4);
    printf("  16 B (4x u32) : %7.1f ns   (%.2fx the 4B cost)\n", t_16, t_16 / t_4);
    printf("  64 B (8x u64) : %7.1f ns   (%.2fx the 4B cost)  <- 1 NoC flit\n", t_64, t_64 / t_4);
    printf(
        "4096 B (loop)   : %7.1f ns   (%.2fx the 4B cost, %.1f ns/64B line)\n",
        t_pay,
        t_pay / t_4,
        t_pay / (PAYLOAD_BYTES / 64));
    printf("=> header poll is ~%.0fx cheaper than a full payload read (sink=%u)\n", t_pay / t_4, sink);
    return 0;
}

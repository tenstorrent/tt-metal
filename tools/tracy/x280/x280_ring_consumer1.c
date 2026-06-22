// X280 SPSC flit-ring consumer — SINGLE-HART BASELINE (no threads, no shared
// state). Establishes the pure per-flit protocol cost before any multi-hart
// optimization. One hart does the whole loop:
//
//   batch == 1 (the bare baseline):  per consumed flit = 3 NoC transactions
//       1. read  w   (NoC read of the producer's write index)
//       2. read  the 64B data flit (8x u64 in one expr = 1 flit transaction)
//       3. write r   (NoC write of the read index, freeing the producer)
//   batch == K:  amortize the w-read and r-write over K flits (drain up to K
//       available cells per w-read / per r-write) to show how much the pointer
//       traffic costs vs the data reads.
//
// The two reads BLOCK the in-order hart (~314 ns NoC round-trip each); the write
// is posted (~free). So batch=1 is expected ~2 reads' worth of latency per flit.
//
// Build on X280: gcc -O2 -o x280_ring_consumer1 x280_ring_consumer1.c
// Run (root):    echo | sudo -S ./x280_ring_consumer1 <tx> <ty> <secs> \
//                  [batch=1] [ring_base=0x80000] [cells=32] [w=0x80800] [r=0x80840] [blk=0x80880]
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
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
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tx> <ty> <secs> [batch] [ring_base] [cells] [w] [r] [blk]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    double secs = atof(argv[3]);
    uint32_t batch = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 1u;
    uint64_t ring_base = argc > 5 ? strtoull(argv[5], 0, 0) : 0x80000UL;
    uint32_t ncells = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 0) : 32u;
    uint64_t w_addr = argc > 7 ? strtoull(argv[7], 0, 0) : 0x80800UL;
    uint64_t r_addr = argc > 8 ? strtoull(argv[8], 0, 0) : 0x80840UL;
    uint64_t blk_addr = argc > 9 ? strtoull(argv[9], 0, 0) : 0x80880UL;
    if (batch < 1 || (ncells & (ncells - 1))) {
        fprintf(stderr, "batch>=1 and cells power of 2\n");
        return 1;
    }

    // Pin to one hart (leave cpu0 for the OS).
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(1, &set);
    sched_setaffinity(0, sizeof set, &set);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    volatile uint32_t* reg = cfg + 0;
    reg[0] = (uint32_t)(ring_base >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (tx & 0x3f) | ((ty & 0x3f) << 6);
    reg[3] = 0;
    volatile uint8_t* win = mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (win == MAP_FAILED) {
        perror("mmap win");
        return 1;
    }
    volatile uint32_t* cells = (volatile uint32_t*)(win + (ring_base & (WINDOW_2M_SIZE - 1)));
    volatile uint32_t* wptr = (volatile uint32_t*)(win + (w_addr & (WINDOW_2M_SIZE - 1)));
    volatile uint32_t* rptr = (volatile uint32_t*)(win + (r_addr & (WINDOW_2M_SIZE - 1)));
    volatile uint32_t* blkptr = (volatile uint32_t*)(win + (blk_addr & (WINDOW_2M_SIZE - 1)));

    const uint32_t mask = ncells - 1;
    uint32_t r = *rptr;  // oldest unread (producer blocks-when-full => [r,w) valid)
    uint32_t start_blocked = *blkptr;
    uint64_t processed = 0, errors = 0;
    uint32_t first_err = 0;
    int had_err = 0;
    uint64_t sink = 0;

    printf("single-hart baseline: Tensix (%u,%u) batch=%u start r=%u\n", tx, ty, batch, r);

    double t0 = now_ns();
    double tend = t0 + secs * 1e9;
    uint64_t since_clock = 0;
    for (;;) {
        uint32_t w = *wptr;  // (1) NoC read of w
        int32_t avail = (int32_t)(w - r);
        if (avail <= 0) {
            // nothing new; spin re-reading w. Bound the clock checks.
            if (++since_clock >= 4096) {
                since_clock = 0;
                if (now_ns() >= tend) {
                    break;
                }
            }
            continue;
        }
        uint32_t n = (uint32_t)avail < batch ? (uint32_t)avail : batch;
        for (uint32_t j = 0; j < n; j++) {
            volatile uint64_t* f = (volatile uint64_t*)(cells + (r & mask) * 16u);  // (2) NoC read of flit
            uint64_t a0 = f[0], a1 = f[1], a2 = f[2], a3 = f[3], a4 = f[4], a5 = f[5], a6 = f[6], a7 = f[7];
            sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
            uint32_t w0 = (uint32_t)a0, w15 = (uint32_t)(a7 >> 32);
            if (w0 != r || w15 != r) {
                if (!had_err) {
                    had_err = 1;
                    first_err = r;
                }
                errors++;
            }
            r++;
        }
        *rptr = r;  // (3) NoC write of r
        processed += n;
        if (++since_clock >= 4096) {
            since_clock = 0;
            if (now_ns() >= tend) {
                break;
            }
        }
    }
    double ns = now_ns() - t0;
    uint32_t final_w = *wptr;
    uint32_t blocked = *blkptr - start_blocked;

    printf("\n============ RING CONSUMER (single hart, batch=%u) ============\n", batch);
    printf("ran %.2f s | processed %lu flits\n", ns / 1e9, processed);
    printf(
        "throughput: %.2f Mflit/s | %.1f ns/flit | %.1f MB/s\n",
        processed * 1e3 / ns,
        ns / processed,
        processed * 64.0 * 1e3 / ns);
    printf(
        "ns per consumed flit breakdown @ batch=%u: includes %s\n",
        batch,
        batch == 1 ? "1 w-read + 1 data-read + 1 r-write" : "amortized w-read/r-write + 1 data-read each");
    printf("producer final w=%u; backpressure events during run: %u\n", final_w, blocked);
    printf("integrity/order errors: %lu", errors);
    if (errors) {
        printf(" (first @ %u)", first_err);
    }
    printf("\n%s\nsink=%lu\n", errors == 0 ? "PASS: lossless + in-order" : "FAIL", (unsigned long)sink);
    return 0;
}

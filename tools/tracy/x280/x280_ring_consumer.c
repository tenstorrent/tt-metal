// X280 SPSC flit-ring consumer (STEP 2, optimized). Drains the Tensix worker's
// L1 flit-ring over the NoC using 3 dedicated harts:
//   - 1 POINTER hart: reads w (NoC), publishes r = min(flusher progress) (NoC).
//   - 2 FLUSHER harts: read the 64B cells (NoC flit reads, 8x u64 per flit) and
//     verify word[0]==word[15]==absolute index (lossless + in-order check).
// The 4th hart is left for Linux (3-worker / 1-OS is the measured sweet spot).
//
// Two optimizations over the naive version (see baseline x280_ring_consumer1.c):
//   1. BATCHED pointer publish: a flusher updates its shared f_next only every
//      PUBLISH_EVERY flits (or right before it has to wait), instead of every
//      flit -- amortizing cross-hart traffic. (Baseline showed per-flit pointer
//      updates cost ~441 ns/flit on one hart.)
//   2. CACHE-LINE SEPARATION: g_w, f_next[0], f_next[1] each live on their own
//      64B line (X280 = 64B lines) so one hart's write doesn't invalidate a line
//      another hart is spinning on (false sharing was throttling the 2nd flusher).
//
// Mechanism: one RW 2MB TLB window mapped to the Tensix core covers the ring
// cells, w and r (all within the same 2MB L1 aperture). Safe Tensix-L1 path.
// Flushers partition indices round-robin (flusher i: base+i, base+i+2, ...); the
// contiguous drained prefix is min(f_next[0], f_next[1]) -> that is r. The
// consumer starts at the producer's current r (oldest unread); because the
// producer blocks-when-full, [r,w) is always valid unread data.
//
// Build on X280:  gcc -O2 -pthread -o x280_ring_consumer x280_ring_consumer.c
// Run (root):     echo | sudo -S ./x280_ring_consumer <tx> <ty> <secs> \
//                   [publish_every=8] [ring_base=0x80000] [cells=32] [w=0x80800] [r=0x80840] [blk=0x80880]
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
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
#define CL 64  // X280 cache-line size

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static inline void cpu_relax(void) { asm volatile("" ::: "memory"); }

// Each independently-written shared word on its own cache line (no false sharing).
typedef struct {
    _Alignas(CL) volatile uint32_t v;
    char pad[CL - sizeof(uint32_t)];
} cl_u32_t;

static cl_u32_t g_w_cl;        // latest w (pointer hart publishes to flushers)
static cl_u32_t f_next_cl[2];  // next absolute index each flusher will read
static cl_u32_t g_stop_cl;     // stop flag

// NoC-mapped pointers and ring geometry.
static volatile uint32_t* g_cells;
static volatile uint32_t* g_wptr;
static volatile uint32_t* g_rptr;
static uint32_t g_ncells;
static uint32_t g_base;
static uint32_t g_publish_every;

// Per-flusher outputs (written only at thread exit -- not hot).
static uint64_t f_processed[2];
static uint64_t f_errors[2];
static uint32_t f_first_err[2];
static volatile uint64_t g_sink;

static pthread_barrier_t g_barrier;

static void pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
}

typedef struct {
    int id;
    int cpu;
} flush_arg_t;

static void* flusher(void* p) {
    flush_arg_t* a = (flush_arg_t*)p;
    pin(a->cpu);
    const int i = a->id;
    const uint32_t mask = g_ncells - 1;
    const uint32_t K = g_publish_every;
    uint32_t idx = g_base + (uint32_t)i;  // round-robin partition
    uint64_t proc = 0, err = 0;
    uint32_t first_err = 0, pending = 0;
    int had_err = 0;
    uint64_t sink = 0;

    pthread_barrier_wait(&g_barrier);
    while (!g_stop_cl.v) {
        if ((int32_t)(idx - g_w_cl.v) >= 0) {
            // About to wait for new cells -- publish progress now so the producer
            // can advance (don't sit on un-published batches while idle).
            if (pending) {
                f_next_cl[i].v = idx;
                pending = 0;
            }
            while ((int32_t)(idx - g_w_cl.v) >= 0) {
                if (g_stop_cl.v) {
                    goto done;
                }
                cpu_relax();
            }
        }
        // Read the whole 64B flit as 8 independent u64 loads (one flit transaction).
        volatile uint64_t* f = (volatile uint64_t*)(g_cells + (idx & mask) * 16u);
        uint64_t a0 = f[0], a1 = f[1], a2 = f[2], a3 = f[3], a4 = f[4], a5 = f[5], a6 = f[6], a7 = f[7];
        sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
        uint32_t w0 = (uint32_t)a0, w15 = (uint32_t)(a7 >> 32);
        if (w0 != idx || w15 != idx) {
            if (!had_err) {
                had_err = 1;
                first_err = idx;
            }
            err++;
        }
        proc++;
        idx += 2;
        if (++pending >= K) {
            f_next_cl[i].v = idx;  // batched publish
            pending = 0;
        }
    }
done:
    f_next_cl[i].v = idx;  // final publish
    f_processed[i] = proc;
    f_errors[i] = err;
    f_first_err[i] = first_err;
    g_sink ^= sink;
    return 0;
}

static void* pointer_hart(void* p) {
    pin(*(int*)p);
    uint32_t last_r = g_base;
    pthread_barrier_wait(&g_barrier);
    while (!g_stop_cl.v) {
        g_w_cl.v = *g_wptr;  // NoC read of producer's write index
        uint32_t a = f_next_cl[0].v, b = f_next_cl[1].v;
        uint32_t r = (a < b) ? a : b;
        if (r != last_r) {
            *g_rptr = r;  // NoC write -> frees the producer
            last_r = r;
        }
        cpu_relax();
    }
    uint32_t a = f_next_cl[0].v, b = f_next_cl[1].v;
    *g_rptr = (a < b) ? a : b;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tx> <ty> <secs> [publish_every] [ring_base] [cells] [w] [r] [blk]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    double secs = atof(argv[3]);
    g_publish_every = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 8u;
    uint64_t ring_base = argc > 5 ? strtoull(argv[5], 0, 0) : 0x80000UL;
    g_ncells = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 0) : 32u;
    uint64_t w_addr = argc > 7 ? strtoull(argv[7], 0, 0) : 0x80800UL;
    uint64_t r_addr = argc > 8 ? strtoull(argv[8], 0, 0) : 0x80840UL;
    uint64_t blk_addr = argc > 9 ? strtoull(argv[9], 0, 0) : 0x80880UL;
    if ((g_ncells & (g_ncells - 1)) != 0 || g_publish_every < 1) {
        fprintf(stderr, "cells power of 2, publish_every >= 1\n");
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
    g_cells = (volatile uint32_t*)(win + (ring_base & (WINDOW_2M_SIZE - 1)));
    g_wptr = (volatile uint32_t*)(win + (w_addr & (WINDOW_2M_SIZE - 1)));
    g_rptr = (volatile uint32_t*)(win + (r_addr & (WINDOW_2M_SIZE - 1)));
    volatile uint32_t* blkptr = (volatile uint32_t*)(win + (blk_addr & (WINDOW_2M_SIZE - 1)));

    g_base = *g_rptr;  // oldest unread (producer never overwrites [r,w))
    g_w_cl.v = *g_wptr;
    f_next_cl[0].v = g_base + 0;
    f_next_cl[1].v = g_base + 1;
    g_stop_cl.v = 0;
    uint32_t start_blocked = *blkptr;

    printf(
        "consumer attaching to Tensix (%u,%u): cells=%u publish_every=%u start r=%u\n",
        tx,
        ty,
        g_ncells,
        g_publish_every,
        g_base);

    pthread_barrier_init(&g_barrier, 0, 4);  // 2 flushers + 1 pointer + main
    pthread_t tp, tf[2];
    int pcpu = 1;
    flush_arg_t fa[2] = {{.id = 0, .cpu = 2}, {.id = 1, .cpu = 3}};
    pthread_create(&tp, 0, pointer_hart, &pcpu);
    pthread_create(&tf[0], 0, flusher, &fa[0]);
    pthread_create(&tf[1], 0, flusher, &fa[1]);

    pthread_barrier_wait(&g_barrier);
    double t0 = now_ns();
    struct timespec ts = {.tv_sec = (time_t)secs, .tv_nsec = (long)((secs - (time_t)secs) * 1e9)};
    nanosleep(&ts, 0);
    g_stop_cl.v = 1;
    double ns = now_ns() - t0;

    pthread_join(tf[0], 0);
    pthread_join(tf[1], 0);
    pthread_join(tp, 0);

    uint64_t processed = f_processed[0] + f_processed[1];
    uint64_t errors = f_errors[0] + f_errors[1];
    uint32_t final_w = *g_wptr;
    uint32_t blocked = *blkptr - start_blocked;
    uint32_t drained_to = (f_next_cl[0].v < f_next_cl[1].v ? f_next_cl[0].v : f_next_cl[1].v);

    printf("\n==================== RING CONSUMER (3 harts, publish_every=%u) ====================\n", g_publish_every);
    printf("ran %.2f s | processed %lu flits (f0=%lu f1=%lu)\n", ns / 1e9, processed, f_processed[0], f_processed[1]);
    printf(
        "throughput: %.2f Mflit/s | %.1f ns/flit | %.1f MB/s\n",
        processed * 1e3 / ns,
        ns / processed,
        processed * 64.0 * 1e3 / ns);
    printf(
        "drained contiguous up to index %u; producer final w=%u (gap=%d)\n",
        drained_to,
        final_w,
        (int32_t)(final_w - drained_to));
    printf("producer backpressure events during run: %u\n", blocked);
    printf("integrity/order errors: %lu", errors);
    if (errors) {
        printf(" (f0 first@%u, f1 first@%u)", f_first_err[0], f_first_err[1]);
    }
    printf(
        "\n%s\nsink=%lu\n",
        errors == 0 ? "PASS: lossless + in-order over NoC" : "FAIL: data loss/reorder detected",
        (unsigned long)g_sink);
    return 0;
}

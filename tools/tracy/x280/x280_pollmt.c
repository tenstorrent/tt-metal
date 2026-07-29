// Multi-threaded uncached NoC-read sweep from the X280, to answer one question:
// does running more harts give more outstanding NoC reads, or does the uncached
// System Port serialize them?
//
// A single in-order X280 hart issuing uncached (System Port) loads is ~1
// outstanding read at a time — that's the latency-bound wall the single-thread
// pollers hit. This spawns N threads (one pinned per hart), each sweeping a
// DISJOINT slice of Tensix cores through its own NoC TLB windows, and reports
// aggregate flit/s. It auto-runs N=1,2,3,4 so you get the scaling curve in one
// shot:
//   - throughput scales toward N  -> the port/NIU overlaps reads; threads ARE
//     the lever (manufacture memory-level parallelism the in-order core can't).
//   - throughput stays flat        -> the System Port serializes; threads don't
//     help and we stop chasing this.
//
// Everything is the uncached System Port (cached is useless for continuous
// monitoring — stale data). Reads each core's 64B flit as 8 independent u64 in
// one expression (the best single-flit pattern from pollall).
//
// usage: ./pollmt <l1_addr> <secs_per_trial> <coordfile> [maxthreads=4] [split]
//   coordfile: lines "CORE <x> <y>" (host launcher prints these)
//   split:     optional; assign window i to NoC (i&1) via noc_sel bit. EXPERIMENTAL
//              — the noc_sel bit position is unvalidated; default is all NoC0.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port, NoC0
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define MAX_CORES 224
#define MAX_THREADS 4
#define NOC_SEL_BIT (1u << 31)  // bit 31 of the window properties word (per ISA docs)

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static volatile uint32_t* g_flit[MAX_CORES];  // per-core 64B flit pointer (shared, read-only)
static int g_ncores;
static volatile int g_stop;
static pthread_barrier_t g_barrier;

typedef struct {
    int lo, hi;              // this thread sweeps cores [lo, hi)
    int cpu;                 // hart to pin to
    unsigned long sweeps;    // out: completed sweeps
    volatile uint32_t sink;  // out: prevent dead-code elimination
} targ_t;

static void* worker(void* p) {
    targ_t* a = (targ_t*)p;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->cpu, &set);
    sched_setaffinity(0, sizeof set, &set);  // pin to our hart

    // Hot loop touches ZERO shared writable memory (local counters only) so the
    // harts don't thrash shared cache lines (false sharing). Write back once.
    uint32_t sink = 0;
    unsigned long sweeps = 0;
    int lo = a->lo, hi = a->hi;
    pthread_barrier_wait(&g_barrier);  // start together with main's timer
    while (!g_stop) {
        for (int i = lo; i < hi; i++) {
            volatile uint64_t* f = (volatile uint64_t*)g_flit[i];
            sink ^= (uint32_t)(f[0] ^ f[1] ^ f[2] ^ f[3] ^ f[4] ^ f[5] ^ f[6] ^ f[7]);
        }
        sweeps++;
    }
    a->sweeps = sweeps;
    a->sink = sink;
    return 0;
}

// Run one trial with nthr threads over all g_ncores cores; return aggregate flit/s.
static double run_trial(int nthr, double secs) {
    g_stop = 0;
    pthread_barrier_init(&g_barrier, 0, nthr + 1);  // +1 for main

    targ_t targ[MAX_THREADS];
    pthread_t tid[MAX_THREADS];
    // Contiguous, balanced partition of cores across threads.
    int base = g_ncores / nthr, extra = g_ncores % nthr, cur = 0;
    for (int t = 0; t < nthr; t++) {
        int cnt = base + (t < extra ? 1 : 0);
        targ[t] = (targ_t){.lo = cur, .hi = cur + cnt, .cpu = t, .sweeps = 0, .sink = 0};
        cur += cnt;
        pthread_create(&tid[t], 0, worker, &targ[t]);
    }

    pthread_barrier_wait(&g_barrier);  // release workers
    double t0 = now_ns();
    struct timespec ts = {.tv_sec = (time_t)secs, .tv_nsec = (long)((secs - (time_t)secs) * 1e9)};
    nanosleep(&ts, 0);
    g_stop = 1;
    double ns = now_ns() - t0;

    unsigned long total_sweeps = 0;
    for (int t = 0; t < nthr; t++) {
        pthread_join(tid[t], 0);
        total_sweeps += targ[t].sweeps;
    }
    pthread_barrier_destroy(&g_barrier);

    // Each sweep of a thread covers its slice; total flits = sum over threads of
    // sweeps*slice. Since slices partition all cores, one "full-grid pass" worth
    // of flits per max(sweeps) isn't uniform — count actual flits directly.
    unsigned long flits = 0;
    {
        int base2 = g_ncores / nthr, extra2 = g_ncores % nthr;
        for (int t = 0; t < nthr; t++) {
            int cnt = base2 + (t < extra2 ? 1 : 0);
            flits += targ[t].sweeps * (unsigned long)cnt;
        }
    }
    double fps = flits * 1e9 / ns;
    double us_per_pass = (double)g_ncores / fps * 1e6;  // time to read all cores once
    printf(
        "  N=%d: %6.2f Mflit/s | %6.2f us per %d-core grid-pass | flits=%lu\n",
        nthr,
        fps / 1e6,
        us_per_pass,
        g_ncores,
        flits);
    return fps;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <l1_addr> <secs_per_trial> <coordfile> [maxthreads=4] [split]\n", argv[0]);
        return 1;
    }
    uint64_t l1_addr = strtoull(argv[1], 0, 0);
    double secs = atof(argv[2]);
    FILE* cf = fopen(argv[3], "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    int maxthr = argc > 4 ? atoi(argv[4]) : MAX_THREADS;
    if (maxthr < 1) {
        maxthr = 1;
    }
    if (maxthr > MAX_THREADS) {
        maxthr = MAX_THREADS;
    }
    int split = (argc > 5 && strcmp(argv[5], "split") == 0);

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
    g_ncores = ncores;

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

    for (int i = 0; i < ncores; i++) {
        volatile uint32_t* reg = cfg + (i * 0x10) / 4;
        reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
        reg[1] = 0;
        reg[2] = (cx[i] & 0x3f) | ((cy[i] & 0x3f) << 6);
        reg[3] = split ? ((i & 1) ? NOC_SEL_BIT : 0) : 0;  // experimental NoC split
        g_flit[i] = (volatile uint32_t*)(wins + (size_t)i * WINDOW_2M_SIZE + (l1_addr & (WINDOW_2M_SIZE - 1)));
    }

    printf(
        "pollmt: %d cores, l1=0x%lx, %.2fs/trial, uncached System Port%s\n",
        ncores,
        l1_addr,
        secs,
        split ? ", NoC split (EXPERIMENTAL)" : ", all NoC0");
    printf("hart-scaling (aggregate throughput vs thread count):\n");

    double base_fps = 0;
    for (int n = 1; n <= maxthr; n++) {
        double fps = run_trial(n, secs);
        if (n == 1) {
            base_fps = fps;
        }
        printf("        -> %.2fx vs N=1\n", base_fps > 0 ? fps / base_fps : 1.0);
    }
    return 0;
}

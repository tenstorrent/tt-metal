// Wake-up latency of a futex-parked thread. A waker on CPU A sleeps for `gap`, stamps the time and wakes a sleeper
// parked on CPU B; the sleeper stamps when it runs again. Also measures a spinning sleeper as the no-park reference.
// Usage: wake_probe <cpu_waker> <cpu_sleeper> [iters]
#define _GNU_SOURCE
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static _Atomic uint32_t word;
static _Atomic int64_t woke_at;
static int spin_mode;
static int iters = 2000;

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

static void* sleeper(void* arg) {
    pin(*(int*)arg);
    for (uint32_t seq = 1; seq <= (uint32_t)iters; seq++) {
        if (spin_mode) {
            while (atomic_load_explicit(&word, memory_order_acquire) < seq) {
            }
        } else {
            uint32_t v;
            while ((v = atomic_load_explicit(&word, memory_order_acquire)) < seq) {
                syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0);
            }
        }
        atomic_store_explicit(&woke_at, now_ns(), memory_order_release);
    }
    return NULL;
}

static int cmp(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x > y) - (x < y);
}

static void run(int cpu_w, int cpu_s, int64_t gap_ns, int spin) {
    spin_mode = spin;
    atomic_store(&word, 0);
    atomic_store(&woke_at, 0);
    pthread_t t;
    pthread_create(&t, NULL, sleeper, &cpu_s);
    pin(cpu_w);
    int64_t* lat = malloc(sizeof(int64_t) * iters);
    for (int i = 0; i < iters; i++) {
        struct timespec ts = {gap_ns / 1000000000, gap_ns % 1000000000};
        nanosleep(&ts, NULL);
        atomic_store(&woke_at, 0);
        int64_t t0 = now_ns();
        atomic_store_explicit(&word, (uint32_t)(i + 1), memory_order_release);
        if (!spin) {
            syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
        }
        int64_t t1;
        while ((t1 = atomic_load_explicit(&woke_at, memory_order_acquire)) == 0) {
        }
        lat[i] = t1 - t0;
    }
    pthread_join(t, NULL);
    qsort(lat, iters, sizeof *lat, cmp);
    printf("WAKE_PROBE mode=%s gap_us=%lld cpus=%d->%d p50_us=%.1f p90_us=%.1f p99_us=%.1f max_us=%.1f\n",
           spin ? "spin" : "futex", (long long)(gap_ns / 1000), cpu_w, cpu_s, lat[iters / 2] / 1e3,
           lat[iters * 9 / 10] / 1e3, lat[iters * 99 / 100] / 1e3, lat[iters - 1] / 1e3);
    free(lat);
}

int main(int argc, char** argv) {
    int cpu_w = argc > 1 ? atoi(argv[1]) : 0;
    int cpu_s = argc > 2 ? atoi(argv[2]) : 1;
    if (argc > 3) {
        iters = atoi(argv[3]);
    }
    run(cpu_w, cpu_s, 100000, 1);
    const int64_t gaps_us[] = {20, 100, 1000, 10000};
    for (unsigned i = 0; i < sizeof gaps_us / sizeof *gaps_us; i++) {
        run(cpu_w, cpu_s, gaps_us[i] * 1000, 0);
    }
    return 0;
}

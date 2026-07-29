// 3-hart continuous 4KB poll of all 110 cores + an export path to the host.
//
// Pollers (harts 0,1,2) sweep their core slice, reading each core's <nbytes>
// over the NoC with the 8-u64-per-flit overlap pattern AND retaining the data
// (storing it into a per-hart SPSC staging ring in X280 DRAM). A 4th thread
// (unpinned, I/O-bound) drains the rings and streams the data over TCP to a host
// sink. Pollers never block on the exporter: if a ring is full they drop (and
// count it) so the poll rate reflects the NoC, decoupled from the slow network.
//
// Measures: poll throughput (NoC), export throughput (network), drop %, and
// whether running the exporter degrades the poll vs the pollalln baseline.
//
// usage: ./pollexport <l1_addr> <secs> <coordfile> <host_ip> <port> [nbytes=4096]
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define MAX_CORES 224
#define NPOLL 3
#define RING_SLOTS 512  // per-hart ring depth (x nbytes)

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static volatile uint64_t* g_base[MAX_CORES];
static int g_ncores, g_flits_per_core, g_slot_bytes;
static volatile int g_stop;
static pthread_barrier_t g_barrier;

// One SPSC ring per poller hart. Producer advances head, consumer advances tail.
typedef struct {
    uint8_t* buf;            // RING_SLOTS * g_slot_bytes
    volatile uint32_t head;  // next slot to write (producer)
    volatile uint32_t tail;  // next slot to read (consumer)
    int lo, hi, cpu;         // core slice + hart
    unsigned long polled;    // out: bytes read from NoC
    unsigned long dropped;   // out: slots dropped (ring full)
} ring_t;

static ring_t g_ring[NPOLL];

static void* poller(void* p) {
    ring_t* r = (ring_t*)p;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(r->cpu, &set);
    sched_setaffinity(0, sizeof set, &set);

    unsigned long polled = 0, dropped = 0;
    int lo = r->lo, hi = r->hi, flits = g_flits_per_core;
    pthread_barrier_wait(&g_barrier);
    while (!g_stop) {
        for (int i = lo; i < hi; i++) {
            uint32_t h = r->head, nh = (h + 1) % RING_SLOTS;
            int full = (nh == r->tail);
            uint64_t* dst = (uint64_t*)(r->buf + (size_t)h * g_slot_bytes);
            volatile uint64_t* w = g_base[i];
            // overlapped 8-u64 flit read, retained into the ring slot
            for (int k = 0; k < flits; k++, w += 8, dst += 8) {
                uint64_t a0 = w[0], a1 = w[1], a2 = w[2], a3 = w[3];
                uint64_t a4 = w[4], a5 = w[5], a6 = w[6], a7 = w[7];
                dst[0] = a0;
                dst[1] = a1;
                dst[2] = a2;
                dst[3] = a3;
                dst[4] = a4;
                dst[5] = a5;
                dst[6] = a6;
                dst[7] = a7;
            }
            polled += g_slot_bytes;
            if (!full) {
                r->head = nh;  // commit
            } else {
                dropped++;  // exporter behind: drop this snapshot
            }
        }
    }
    r->polled = polled;
    r->dropped = dropped;
    return 0;
}

static int sendall(int fd, const uint8_t* b, size_t n) {
    while (n) {
        ssize_t s = send(fd, b, n, 0);
        if (s <= 0) {
            return -1;
        }
        b += s;
        n -= (size_t)s;
    }
    return 0;
}

static unsigned long g_sent;  // bytes exported

static void* exporter(void* arg) {
    int fd = *(int*)arg;
    unsigned long sent = 0;
    pthread_barrier_wait(&g_barrier);
    while (1) {
        int any = 0;
        for (int t = 0; t < NPOLL; t++) {
            ring_t* r = &g_ring[t];
            while (r->tail != r->head) {
                if (sendall(fd, r->buf + (size_t)r->tail * g_slot_bytes, g_slot_bytes) < 0) {
                    goto done;
                }
                r->tail = (r->tail + 1) % RING_SLOTS;
                sent += g_slot_bytes;
                any = 1;
            }
        }
        if (!any && g_stop) {
            break;  // drained after stop
        }
    }
done:
    g_sent = sent;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <l1_addr> <secs> <coordfile> <host_ip> <port> [nbytes=4096]\n", argv[0]);
        return 1;
    }
    uint64_t l1_addr = strtoull(argv[1], 0, 0);
    double secs = atof(argv[2]);
    FILE* cf = fopen(argv[3], "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    const char* ip = argv[4];
    int port = atoi(argv[5]);
    uint32_t nbytes = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 0) : 4096;
    g_slot_bytes = nbytes;
    g_flits_per_core = nbytes / 64;

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
    g_ncores = ncores;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    volatile uint8_t* wins = mmap(0, (size_t)ncores * WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (cfg == MAP_FAILED || wins == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    for (int i = 0; i < ncores; i++) {
        volatile uint32_t* reg = cfg + (i * 0x10) / 4;
        reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
        reg[1] = 0;
        reg[2] = (cx[i] & 0x3f) | ((cy[i] & 0x3f) << 6);
        reg[3] = 0;
        g_base[i] = (volatile uint64_t*)(wins + (size_t)i * WINDOW_2M_SIZE + (l1_addr & (WINDOW_2M_SIZE - 1)));
    }

    // TCP connection to the host sink for export.
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, ip, &a.sin_addr);
    if (connect(sock, (void*)&a, sizeof a)) {
        perror("connect host sink");
        return 1;
    }

    // Partition cores across NPOLL harts; allocate per-hart rings.
    int base = ncores / NPOLL, extra = ncores % NPOLL, cur = 0;
    for (int t = 0; t < NPOLL; t++) {
        int cnt = base + (t < extra ? 1 : 0);
        g_ring[t] = (ring_t){0};
        g_ring[t].buf = malloc((size_t)RING_SLOTS * g_slot_bytes);
        g_ring[t].lo = cur;
        g_ring[t].hi = cur + cnt;
        g_ring[t].cpu = t;
        cur += cnt;
    }

    pthread_barrier_init(&g_barrier, 0, NPOLL + 2);  // 3 pollers + exporter + main
    pthread_t pt[NPOLL], et;
    for (int t = 0; t < NPOLL; t++) {
        pthread_create(&pt[t], 0, poller, &g_ring[t]);
    }
    pthread_create(&et, 0, exporter, &sock);

    pthread_barrier_wait(&g_barrier);
    double t0 = now_ns();
    struct timespec ts = {.tv_sec = (time_t)secs, .tv_nsec = (long)((secs - (time_t)secs) * 1e9)};
    nanosleep(&ts, 0);
    g_stop = 1;
    double poll_ns = now_ns() - t0;

    unsigned long polled = 0, dropped = 0, slots_total = 0;
    for (int t = 0; t < NPOLL; t++) {
        pthread_join(pt[t], 0);
        polled += g_ring[t].polled;
        dropped += g_ring[t].dropped;
    }
    slots_total = polled / g_slot_bytes;
    pthread_join(et, 0);
    close(sock);

    double poll_mbps = polled / 1e6 / (poll_ns / 1e9);
    double exp_mbps = g_sent / 1e6 / (poll_ns / 1e9);
    printf("3-hart poll + export: %d cores x %uB, %.1fs\n", ncores, nbytes, poll_ns / 1e9);
    printf(
        "  POLL  : %.1f MB/s  (%lu snapshots, %.2f us per %d-core grid-pass)\n",
        poll_mbps,
        slots_total / ncores,
        poll_ns / 1e3 / (slots_total / ncores),
        ncores);
    printf("  EXPORT: %.1f MB/s  (%lu MB sent over TCP)\n", exp_mbps, g_sent / 1000000);
    printf(
        "  DROP  : %.1f%% (%lu of %lu snapshots dropped, ring full)\n",
        100.0 * dropped / (dropped + g_sent / g_slot_bytes),
        dropped,
        slots_total);
    return 0;
}

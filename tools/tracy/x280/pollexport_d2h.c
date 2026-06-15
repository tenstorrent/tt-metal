// 3-hart continuous 4KB poll of all 110 cores + export to host through the
// PCIe-NoC D2H socket (the fast path), replacing pollexport.c's slow SLIRP TCP
// exporter.
//
// Pollers (harts 0,1,2) sweep their core slice, reading each core's <nbytes>
// over the NoC with the 8-u64-per-flit overlap pattern and retaining the data
// into a per-hart SPSC staging ring in X280 DRAM. A 4th (unpinned) exporter
// thread drains the rings and pushes each slot into the host D2HSocket FIFO via
// a PCIe-tile TLB window, with the real socket flow control: it advances
// write_ptr (wrapping the FIFO), bumps bytes_sent (NoC-written to the host
// bytes_sent word), and throttles on bytes_acked (NoC-read from the Tensix
// sender core's config buffer L1, which the host's read() updates). Pollers
// never block on the exporter: if a ring is full they drop and count it.
//
// Measures: poll throughput (NoC read), export throughput (PCIe write), drop %.
// Compare to the SLIRP baseline (pollexport.c): poll 326 / export 14.5 / 95.5%.
//
// Host side (drains the FIFO, fifo/page must match the args here):
//   env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_SKIP_DRAM_TLBS=1 \
//     ./build_Release/programming_examples/metal_example_x280_d2h 3 serve <fifo> <page> <secs>
//
// usage (run as root on the X280):
//   ./pollexport_d2h <l1_addr> <secs> <coordfile> <tensix_x> <tensix_y> <config_addr> \
//                    [nbytes=4096] [fifo=0x10000] [page=4096] [pcie_x=2] [pcie_y=0]
//   nbytes (poll size/core) should equal page (D2H page size) for 1 slot = 1 page.

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
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define WINDOW_2M_MASK (WINDOW_2M_SIZE - 1)
#define MAX_CORES 224
#define NPOLL 3
#define RING_SLOTS 512

// poller windows are 0..ncores-1; the exporter uses the next two indices
// (set at runtime from ncores). Data + bytes_sent share WIN_PCIE (same 2MB page
// on the PCIe tile); WIN_CFG reads the Tensix sender config L1 (host_data addr,
// fifo size, and the live bytes_acked counter for flow control).
static int WIN_PCIE, WIN_CFG;

// D2H sender config-buffer word offsets (tt_metal/hw/inc/hostdev/socket.h)
#define W_BSENT_LO 3
#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_IS_D2H 6
#define W_BACKED 8  // bytes_acked[0] at md_size(32)/4
#define W_BSENT_HI 12
#define W_DATA_HI 13
#define W_PCIE_ENC 14

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static volatile uint64_t* g_base[MAX_CORES];
static int g_flits_per_core, g_slot_bytes;
static volatile int g_stop;
static pthread_barrier_t g_barrier;

typedef struct {
    uint8_t* buf;
    volatile uint32_t head, tail;
    int lo, hi, cpu;
    unsigned long polled, dropped;
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
                r->head = nh;
            } else {
                dropped++;
            }
        }
    }
    r->polled = polled;
    r->dropped = dropped;
    return 0;
}

// --- exporter: drain rings -> host D2H FIFO over PCIe, with flow control ---
static uint64_t g_host_data, g_host_bsent;
static uint32_t g_fifo_size, g_page_size, g_pcie_enc_unused;
static volatile uint8_t* g_dwin;   // WIN_DATA data region (RW, PCIe tile)
static volatile uint32_t* g_bsp;   // bytes_sent word in WIN_BSENT
static volatile uint32_t* g_ackp;  // bytes_acked word in WIN_ACK (Tensix L1)
static unsigned long g_sent;
static unsigned long g_stall_spins;

static void* exporter(void* arg) {
    (void)arg;
    uint32_t write_ptr = 0, bytes_sent = 0, bytes_acked = 0;
    unsigned long sent = 0, spins = 0;
    const uint32_t page = g_page_size;
    pthread_barrier_wait(&g_barrier);
    while (1) {
        int any = 0;
        for (int t = 0; t < NPOLL; t++) {
            ring_t* r = &g_ring[t];
            while (r->tail != r->head) {
                // flow control: wait until the FIFO has room for one page
                while ((uint32_t)(bytes_sent - bytes_acked) > g_fifo_size - page) {
                    bytes_acked = *g_ackp;  // NoC read of host-updated ack
                    spins++;
                    if (g_stop) {
                        goto done;
                    }
                }
                // write one page from the ring slot into the host FIFO
                volatile uint64_t* d = (volatile uint64_t*)(g_dwin + ((g_host_data + write_ptr) & WINDOW_2M_MASK));
                const uint64_t* s = (const uint64_t*)(r->buf + (size_t)r->tail * g_slot_bytes);
                for (uint32_t b = 0; b < page; b += 64, d += 8, s += 8) {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = s[3];
                    d[4] = s[4];
                    d[5] = s[5];
                    d[6] = s[6];
                    d[7] = s[7];
                }
                volatile uint64_t fence = d[-1];  // drain posted writes before notify
                (void)fence;
                write_ptr = (write_ptr + page) % g_fifo_size;
                bytes_sent += page;
                *g_bsp = bytes_sent;  // notify host (NoC write of bytes_sent)
                r->tail = (r->tail + 1) % RING_SLOTS;
                sent += page;
                any = 1;
            }
        }
        if (!any && g_stop) {
            break;
        }
    }
done:
    g_sent = sent;
    g_stall_spins = spins;
    return 0;
}

static void program_window(volatile uint32_t* cfg, int win, uint64_t noc_addr, unsigned x, unsigned y) {
    volatile uint32_t* reg = cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(noc_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(noc_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    reg[3] = 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(
            stderr,
            "usage: %s <l1_addr> <secs> <coordfile> <tensix_x> <tensix_y> <config_addr> "
            "[nbytes=4096] [fifo=0x10000] [page=4096] [pcie_x=2] [pcie_y=0]\n",
            argv[0]);
        return 1;
    }
    uint64_t l1_addr = strtoull(argv[1], 0, 0);
    double secs = atof(argv[2]);
    FILE* cf = fopen(argv[3], "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    unsigned tx = atoi(argv[4]), ty = atoi(argv[5]);
    uint64_t cfg_addr = strtoull(argv[6], 0, 0);
    uint32_t nbytes = argc > 7 ? (uint32_t)strtoul(argv[7], 0, 0) : 4096;
    uint32_t fifo = argc > 8 ? (uint32_t)strtoul(argv[8], 0, 0) : 0x10000;
    uint32_t page = argc > 9 ? (uint32_t)strtoul(argv[9], 0, 0) : 4096;
    unsigned px = argc > 10 ? atoi(argv[10]) : 2;
    unsigned py = argc > 11 ? atoi(argv[11]) : 0;
    g_slot_bytes = nbytes;
    g_flits_per_core = nbytes / 64;
    g_fifo_size = fifo;
    g_page_size = page;

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

    // The X280 has ~110 small TLB windows (indices 0..109); higher indices wrap
    // and alias low ones. We can't grab a "spare" high window for the exporter,
    // so reserve the LAST window for the PCIe write and poll one fewer core. The
    // bytes_acked flow-control read reuses poller window 0 (it maps the sender
    // core (1,2) and its 2MB page also covers the config buffer at cfg_addr).
    int npoll = ncores - 1;  // cores actually polled (windows 0..npoll-1)
    WIN_PCIE = npoll;        // last genuine window, RW -> PCIe tile (data + bytes_sent)
    WIN_CFG = 0;             // (unused; ack reuses poller window 0)

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    volatile uint32_t* cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    volatile uint8_t* wins = mmap(0, (size_t)npoll * WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (cfg == MAP_FAILED || wins == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    for (int i = 0; i < npoll; i++) {
        program_window(cfg, i, l1_addr, cx[i], cy[i]);
        g_base[i] = (volatile uint64_t*)(wins + (size_t)i * WINDOW_2M_SIZE + (l1_addr & WINDOW_2M_MASK));
    }

    // Read the D2H socket config from the sender core L1 via poller window 0.
    // Requires core 0 == sender (1,2) and l1_addr/cfg_addr in the same 2MB page.
    if (cx[0] != tx || cy[0] != ty || (l1_addr >> WINDOW_2M_SHIFT) != (cfg_addr >> WINDOW_2M_SHIFT)) {
        fprintf(
            stderr,
            "FATAL: need coordfile[0]==sender (%u,%u) and l1_addr/cfg_addr same 2MB page "
            "(got core0=(%u,%u), l1=0x%lx cfg=0x%lx)\n",
            tx,
            ty,
            cx[0],
            cy[0],
            l1_addr,
            cfg_addr);
        return 2;
    }
    volatile uint32_t* c = (volatile uint32_t*)(wins + (cfg_addr & WINDOW_2M_MASK));
    g_host_data = ((uint64_t)c[W_DATA_HI] << 32) | c[W_FIFO_LO];
    g_host_bsent = ((uint64_t)c[W_BSENT_HI] << 32) | c[W_BSENT_LO];
    uint32_t cfg_fifo = c[W_FIFO_SZ];
    uint32_t cfg_is_d2h = c[W_IS_D2H];
    g_ackp = (volatile uint32_t*)(wins + ((cfg_addr + 4 * W_BACKED) & WINDOW_2M_MASK));

    printf(
        "D2H stream: polling %d/%d cores x %uB, fifo=%u page=%u, host_data=0x%lx (host fifo=%u is_d2h=%u), PCIe "
        "(%u,%u)\n",
        npoll,
        ncores,
        nbytes,
        fifo,
        page,
        g_host_data,
        cfg_fifo,
        cfg_is_d2h,
        px,
        py);
    if (!cfg_is_d2h || fifo != cfg_fifo) {
        fprintf(
            stderr,
            "FATAL: config read failed or fifo mismatch (is_d2h=%u, host fifo=%u, arg fifo=%u)\n",
            cfg_is_d2h,
            cfg_fifo,
            fifo);
        return 2;
    }
    if (nbytes != page) {
        fprintf(stderr, "note: nbytes(%u) != page(%u); 1 ring slot != 1 D2H page\n", nbytes, page);
    }
    // data + bytes_sent must both sit within one 2MB window from host_data's page base
    if ((g_host_data & WINDOW_2M_MASK) + fifo + 4 > WINDOW_2M_SIZE) {
        fprintf(
            stderr,
            "FATAL: fifo+bytes_sent crosses a 2MB window (host_data off 0x%lx); use smaller fifo / re-run\n",
            g_host_data & WINDOW_2M_MASK);
        return 2;
    }

    // single RW window to the PCIe tile for both data and bytes_sent
    g_dwin = mmap(
        0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + (size_t)WIN_PCIE * WINDOW_2M_SIZE);
    if (g_dwin == MAP_FAILED) {
        perror("mmap pcie win");
        return 1;
    }
    program_window(cfg, WIN_PCIE, g_host_data, px, py);
    g_bsp = (volatile uint32_t*)(g_dwin + (g_host_bsent & WINDOW_2M_MASK));

    // Read-probe the PCIe write window before any write: the host FIFO is
    // zero-initialized, so host_data[0] must read back 0. This confirms WIN_PCIE
    // is a valid window AND that the path to the PCIe tile is live, before we
    // ever issue a write (a malformed write to a bad target could hang the NoC).
    volatile uint32_t probe = *(volatile uint32_t*)(g_dwin + (g_host_data & WINDOW_2M_MASK));
    printf("pre-stream probe: host_data[0] via PCIe win %d = 0x%08x (expect 0)\n", WIN_PCIE, probe);

    int base = npoll / NPOLL, extra = npoll % NPOLL, cur = 0;
    for (int t = 0; t < NPOLL; t++) {
        int cnt = base + (t < extra ? 1 : 0);
        g_ring[t] = (ring_t){0};
        g_ring[t].buf = malloc((size_t)RING_SLOTS * g_slot_bytes);
        g_ring[t].lo = cur;
        g_ring[t].hi = cur + cnt;
        g_ring[t].cpu = t;
        cur += cnt;
    }

    pthread_barrier_init(&g_barrier, 0, NPOLL + 2);
    pthread_t pt[NPOLL], et;
    for (int t = 0; t < NPOLL; t++) {
        pthread_create(&pt[t], 0, poller, &g_ring[t]);
    }
    pthread_create(&et, 0, exporter, 0);

    pthread_barrier_wait(&g_barrier);
    double t0 = now_ns();
    struct timespec ts = {.tv_sec = (time_t)secs, .tv_nsec = (long)((secs - (time_t)secs) * 1e9)};
    nanosleep(&ts, 0);
    g_stop = 1;
    double poll_ns = now_ns() - t0;

    unsigned long polled = 0, dropped = 0;
    for (int t = 0; t < NPOLL; t++) {
        pthread_join(pt[t], 0);
        polled += g_ring[t].polled;
        dropped += g_ring[t].dropped;
    }
    pthread_join(et, 0);

    unsigned long slots_total = polled / g_slot_bytes;
    unsigned long snaps = slots_total / npoll;
    double poll_mbps = polled / 1e6 / (poll_ns / 1e9);
    double exp_mbps = g_sent / 1e6 / (poll_ns / 1e9);
    printf("3-hart poll + D2H export: %.2fs\n", poll_ns / 1e9);
    printf(
        "  POLL  : %.1f MB/s  (%lu snapshots, %.2f us per %d-core grid-pass)\n",
        poll_mbps,
        snaps,
        snaps ? poll_ns / 1e3 / snaps : 0.0,
        npoll);
    printf(
        "  EXPORT: %.1f MB/s  (%lu MB pushed to host over PCIe, %lu flow-control spins)\n",
        exp_mbps,
        g_sent / 1000000,
        g_stall_spins);
    printf(
        "  DROP  : %.1f%% (%lu of %lu poll-slots dropped, ring full)\n",
        100.0 * dropped / (dropped + slots_total ? (dropped + slots_total) : 1),
        dropped,
        dropped + slots_total);
    return 0;
}

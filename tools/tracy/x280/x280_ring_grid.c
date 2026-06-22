// X280 SPSC flit-ring GRID consumer (STEP 3) -- ONE process per hart, each
// owning a disjoint COLUMN-BAND of the Tensix grid. This is the scaling axis the
// measurements pointed to: piling harts on one tile is NIU-contention-bound
// (~2 Mflit/s), but harts on DIFFERENT tiles scale ~linearly (different NIUs +
// disjoint NoC routes -> the ~430 MB/s 3-hart grid ceiling). Each core runs an
// independent flit-ring producer in its own L1; this process drains the cores in
// its column-band with ZERO shared state (no pthreads, no false sharing).
//
// Partition: the cores are split by NoC X into `nregions` contiguous column
// groups; this process handles group `region_id`. Each region uses a DISJOINT
// block of TLB windows ([region_id*WIN_STRIDE ...]) so 3 processes don't collide
// on the shared window config registers.
//
// Run 3 instances pinned to harts 1/2/3 (leave hart 0 for Linux):
//   for r in 0 1 2; do taskset -c $((r+1)) ./x280_ring_grid <secs> 3 $r coords.txt & done; wait
//
// Build on X280: gcc -O2 -o x280_ring_grid x280_ring_grid.c
#define _GNU_SOURCE
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
#define MAX_CORES 256
#define WIN_STRIDE 74  // windows reserved per region (3*74=222 < 224 available)

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

typedef struct {
    unsigned x, y;
    volatile uint32_t* cells;  // ring base for this core (windowed)
    volatile uint32_t* wptr;
    volatile uint32_t* rptr;
    uint32_t r;  // local read index for this core
} core_t;

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(
            stderr, "usage: %s <secs> <nregions> <region_id> <coordfile> [ring_base] [cells] [w] [r] [blk]\n", argv[0]);
        return 1;
    }
    double secs = atof(argv[1]);
    int nregions = atoi(argv[2]);
    int region_id = atoi(argv[3]);
    const char* coordfile = argv[4];
    uint64_t ring_base = argc > 5 ? strtoull(argv[5], 0, 0) : 0x80000UL;
    uint32_t ncells = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 0) : 32u;
    uint64_t w_addr = argc > 7 ? strtoull(argv[7], 0, 0) : 0x80800UL;
    uint64_t r_addr = argc > 8 ? strtoull(argv[8], 0, 0) : 0x80840UL;
    (void)argc;
    if (region_id < 0 || region_id >= nregions || (ncells & (ncells - 1))) {
        fprintf(stderr, "bad region or cells\n");
        return 1;
    }

    // Parse all cores.
    unsigned ax[MAX_CORES], ay[MAX_CORES];
    int nall = 0;
    FILE* cf = fopen(coordfile, "r");
    if (!cf) {
        perror("coordfile");
        return 1;
    }
    char line[128];
    unsigned minx = ~0u, maxx = 0;
    while (fgets(line, sizeof line, cf) && nall < MAX_CORES) {
        unsigned x, y;
        if (sscanf(line, "CORE %u %u", &x, &y) == 2) {
            ax[nall] = x;
            ay[nall] = y;
            if (x < minx) {
                minx = x;
            }
            if (x > maxx) {
                maxx = x;
            }
            nall++;
        }
    }
    fclose(cf);
    if (nall == 0) {
        fprintf(stderr, "no cores parsed\n");
        return 1;
    }

    // Column-band partition: split the X range [minx,maxx] into nregions bands;
    // this process owns cores whose X falls in band region_id.
    unsigned span = maxx - minx + 1;
    unsigned lo = minx + (uint64_t)span * region_id / nregions;
    unsigned hi = minx + (uint64_t)span * (region_id + 1) / nregions;  // exclusive

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
    // Map this region's disjoint window block (WIN_STRIDE windows starting at an
    // offset unique to region_id). Each core gets one 2MB window.
    size_t win_base_idx = (size_t)region_id * WIN_STRIDE;
    volatile uint8_t* wins = mmap(
        0,
        (size_t)WIN_STRIDE * WINDOW_2M_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        WINDOW_2M_BASE + win_base_idx * WINDOW_2M_SIZE);
    if (wins == MAP_FAILED) {
        perror("mmap wins");
        return 1;
    }

    core_t cores[MAX_CORES];
    int n = 0;
    for (int i = 0; i < nall; i++) {
        if (ax[i] < lo || ax[i] >= hi) {
            continue;
        }
        if (n >= WIN_STRIDE) {
            fprintf(stderr, "region %d has >%d cores\n", region_id, WIN_STRIDE);
            return 1;
        }
        size_t wi = win_base_idx + n;  // absolute window index
        volatile uint32_t* reg = cfg + (wi * 0x10) / 4;
        reg[0] = (uint32_t)(ring_base >> WINDOW_2M_SHIFT);
        reg[1] = 0;
        reg[2] = (ax[i] & 0x3f) | ((ay[i] & 0x3f) << 6);
        reg[3] = 0;
        volatile uint8_t* base = wins + (size_t)n * WINDOW_2M_SIZE;
        cores[n].x = ax[i];
        cores[n].y = ay[i];
        cores[n].cells = (volatile uint32_t*)(base + (ring_base & (WINDOW_2M_SIZE - 1)));
        cores[n].wptr = (volatile uint32_t*)(base + (w_addr & (WINDOW_2M_SIZE - 1)));
        cores[n].rptr = (volatile uint32_t*)(base + (r_addr & (WINDOW_2M_SIZE - 1)));
        cores[n].r = *cores[n].rptr;  // start at oldest unread (blocking => [r,w) valid)
        n++;
    }
    if (n == 0) {
        fprintf(stderr, "region %d owns no cores (X band [%u,%u))\n", region_id, lo, hi);
        return 1;
    }

    const uint32_t mask = ncells - 1;
    uint64_t processed = 0, errors = 0;
    uint32_t first_err_core = 0, first_err_idx = 0;
    int had_err = 0;
    uint64_t sink = 0;

    printf(
        "region %d: hart owns %d cores, X band [%u,%u), windows [%zu,%zu)\n",
        region_id,
        n,
        lo,
        hi,
        win_base_idx,
        win_base_idx + n);

    double t0 = now_ns();
    double tend = t0 + secs * 1e9;
    uint64_t since_clock = 0;
    for (;;) {
        // Round-robin across this region's cores. Cross-tile visits give the
        // latency overlap a single tile can't (independent NIUs).
        for (int c = 0; c < n; c++) {
            uint32_t w = *cores[c].wptr;  // NoC read of this core's w
            uint32_t r = cores[c].r;
            int32_t avail = (int32_t)(w - r);
            if (avail <= 0) {
                continue;
            }
            for (int32_t j = 0; j < avail; j++) {
                volatile uint64_t* f = (volatile uint64_t*)(cores[c].cells + (r & mask) * 16u);
                uint64_t a0 = f[0], a1 = f[1], a2 = f[2], a3 = f[3], a4 = f[4], a5 = f[5], a6 = f[6], a7 = f[7];
                sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
                uint32_t w0 = (uint32_t)a0, w15 = (uint32_t)(a7 >> 32);
                if (w0 != r || w15 != r) {
                    if (!had_err) {
                        had_err = 1;
                        first_err_core = c;
                        first_err_idx = r;
                    }
                    errors++;
                }
                r++;
            }
            cores[c].r = r;
            *cores[c].rptr = r;  // publish (per core, once per visit)
            processed += avail;
        }
        if (++since_clock >= 256) {
            since_clock = 0;
            if (now_ns() >= tend) {
                break;
            }
        }
    }
    double ns = now_ns() - t0;

    printf("\n---- region %d done: %d cores ----\n", region_id, n);
    printf(
        "processed %lu flits in %.2f s | %.2f Mflit/s | %.1f MB/s | %.1f ns/flit\n",
        processed,
        ns / 1e9,
        processed * 1e3 / ns,
        processed * 64.0 * 1e3 / ns,
        ns / processed);
    printf("integrity/order errors: %lu", errors);
    if (errors) {
        printf(" (first core idx %u @ ring index %u)", first_err_core, first_err_idx);
    }
    printf("\n%s\nsink=%lu\n", errors == 0 ? "PASS region" : "FAIL region", (unsigned long)sink);
    return 0;
}

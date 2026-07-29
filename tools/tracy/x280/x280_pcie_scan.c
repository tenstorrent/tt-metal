// SAFE read-only scan to find how the X280 must address host memory through the
// PCIe tile. All operations are NoC READS (a read cannot hang the NoC the way a
// malformed write can), so this is the low-risk way to nail down the correct
// (coordinate, address-bits) before doing any write.
//
// Oracle: run the host in `hold` mode first -- it pins a zero-initialized FIFO
// and does NOT write it. So a read that REACHES host memory returns 0x00000000;
// a read that misses (wrong coord / wrong NoC-to-Host window / unmapped IOVA)
// returns 0xffffffff (PCIe master-abort fill). We scan coordinates x NoC-to-Host
// window-select bits and print what each returns; the winning combo reads 0.
//
// ISA facts (BlackholeA0/PCIExpressTile + L2CPUTile/TLBWindows.md):
//  - The PCIe tile's NoC-to-Host space is 64 windows selected by NoC addr bits
//    [63:58]: 0x00-0x0F route DIRECT to the host IOMMU; 0x10-0x1F route through
//    the outbound iATU. So bits[57:0] are the host IOVA when winsel in 0x00-0x0F.
//  - The X280 small TLB window forms target = local_offset[42:0]<<21 | off[20:0],
//    so it can express all 64 target bits incl. the [63:58] window-select.
//  - metal's D2H socket uses winsel=0 (plain IOVA) and the PCIe tile's TRANSLATED
//    coord (19,24); the X280 pollers address Tensix by PHYSICAL coord. Which
//    coordinate the PCIe tile wants from an X280-sourced read is what we test.
//
// usage (root): ./x280_pcie_scan <tensix_x> <tensix_y> <config_l1_addr>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)  // uncached System Port
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)
#define WINDOW_2M_MASK (WINDOW_2M_SIZE - 1)

#define W_FIFO_LO 4
#define W_FIFO_SZ 5
#define W_IS_D2H 6
#define W_DATA_HI 13
#define W_PCIE_ENC 14

static volatile uint32_t* g_cfg;
static int g_fd;

static void program_small(int win, uint64_t target, unsigned x, unsigned y) {
    volatile uint32_t* reg = g_cfg + (win * 0x10) / 4;
    uint64_t local_offset = target >> WINDOW_2M_SHIFT;  // bits[42:0]
    reg[0] = (uint32_t)(local_offset & 0xFFFFFFFF);
    reg[1] = (uint32_t)(local_offset >> 32);
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);  // noc_properties_lo: x_end[5:0], y_end[11:6]
    reg[3] = 0;                               // default ordering, NoC0
}

// read one u32 at host IOVA `host_addr` via NoC-to-host winsel through PCIe tile
// (x,y), using X280 small window index `win`.
static uint32_t probe_read(int win, unsigned x, unsigned y, uint64_t host_addr, unsigned winsel) {
    uint64_t target = ((uint64_t)winsel << 58) | host_addr;
    program_small(win, target, x, y);
    volatile uint8_t* w =
        mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, g_fd, WINDOW_2M_BASE + (size_t)win * WINDOW_2M_SIZE);
    if (w == MAP_FAILED) {
        return 0xDEAD0001;
    }
    volatile uint32_t* p = (volatile uint32_t*)(w + (target & WINDOW_2M_MASK));
    uint32_t v = p[0];
    munmap((void*)w, WINDOW_2M_SIZE);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tensix_x> <tensix_y> <config_l1_addr>\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);

    g_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    g_cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, TLB_2M_CONFIG_BASE);

    // read the socket config from the Tensix sender L1 (window 0, plain coord)
    program_small(0, cfg_addr, tx, ty);
    volatile uint8_t* w0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, g_fd, WINDOW_2M_BASE);
    volatile uint32_t* c = (volatile uint32_t*)(w0 + (cfg_addr & WINDOW_2M_MASK));
    uint64_t host_data = ((uint64_t)c[W_DATA_HI] << 32) | c[W_FIFO_LO];
    uint32_t fifo_sz = c[W_FIFO_SZ], is_d2h = c[W_IS_D2H], pcie_enc = c[W_PCIE_ENC];
    unsigned tenc_x = pcie_enc & 0x3f, tenc_y = (pcie_enc >> 6) & 0x3f;
    munmap((void*)w0, WINDOW_2M_SIZE);

    printf(
        "config @ (%u,%u):0x%lx -> host_data=0x%lx fifo=%u is_d2h=%u pcie_enc=0x%x (translated %u,%u)\n",
        tx,
        ty,
        cfg_addr,
        host_data,
        fifo_sz,
        is_d2h,
        pcie_enc,
        tenc_x,
        tenc_y);
    if (!is_d2h) {
        fprintf(stderr, "no live D2H socket; start host `hold` first\n");
        return 2;
    }
    printf("oracle: host FIFO is zero-init in `hold` mode -> 0x00000000 = reached host, 0xffffffff = missed\n\n");

    struct {
        unsigned x, y;
        const char* tag;
    } coords[] = {
        {2, 0, "phys PCIe0 (2,0)"},
        {11, 0, "phys PCIe1 (11,0)"},
        {tenc_x, tenc_y, "translated (from enc)"},
        {19, 24, "translated PCIe0 (19,24)"},
    };
    // ONLY direct-IOMMU window-selects (0x00-0x0F). winsel >= 0x10 routes via the
    // outbound iATU; a READ into an unconfigured iATU window stalls the NIU
    // forever and wedges the hart + the X280's sshd (observed 2026-06-12 on
    // bh-qb-05). NEVER scan winsel >= 0x10. 0,1,2 are plain IOVA variants; 4 sets
    // bit60 (the "non-iATU direct 64-bit" path the writer uses).
    unsigned winsels[] = {0, 1, 2, 4};

    printf("%-26s", "coord \\ winsel");
    for (unsigned i = 0; i < sizeof winsels / sizeof *winsels; i++) {
        printf("  ws=0x%-6x", winsels[i]);
    }
    printf("\n");
    for (unsigned ci = 0; ci < sizeof coords / sizeof *coords; ci++) {
        printf("%-26s", coords[ci].tag);
        for (unsigned wi = 0; wi < sizeof winsels / sizeof *winsels; wi++) {
            uint32_t v = probe_read(1, coords[ci].x, coords[ci].y, host_data, winsels[wi]);
            printf("  %08x  ", v);
            fflush(stdout);
        }
        printf("\n");
    }
    printf("\n(a column/row reading 00000000 is the correct host-addressing mode)\n");
    return 0;
}

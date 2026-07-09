// X280 -> HOST write through the PCIe NoC tile, into a tt-metal D2HSocket.
//
// This is the fast export path: the X280 NoC-writes profiler data straight to
// host pinned memory through the chip's PCIe tile, bypassing both the slow
// SLIRP network and the (Linux-occupied) device DRAM. It replicates what the
// Tensix sender kernel pcie_socket_sender.cpp does, but using X280 TLB-window
// stores -- the X280 has no NIU command-buffer path (disproven 2026-06-11), so
// memory-mapped loads/stores through a TLB window are its ONLY NoC mechanism.
//
// Sequence:
//   1. Read the 64-byte D2H socket config buffer from the Tensix sender core's
//      L1 over a NoC read window. Decode the host data-FIFO addr, the host
//      bytes_sent addr, and pcie_xy_enc (see tt_metal/hw/inc/hostdev/socket.h).
//   2. Program a READ+WRITE 2MB window to the PCIe tile (physical NOC0 (2,0) on
//      this 4-chip box) and store <nbytes> of a known pattern to the host data
//      FIFO addr. With IOMMU on, the host IOVA is < 4 GB, so a single 2MB window
//      reaches both the data and the bytes_sent word (no 128GB window needed).
//   3. Read-back fence (forces the posted data writes to drain), then write
//      bytes_sent = nbytes to the host bytes_sent addr. That release lets the
//      host D2HSocket::read() return the page.
//
// Host side (metal, bh container), prints the Tensix coord + config addr:
//   env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_SKIP_DRAM_TLBS=1 \
//     ./build_Release/programming_examples/metal_example_x280_d2h 3 listen 1 120
//
// usage (run as root on the X280, needs /dev/mem):
//   ./x280_d2h_write <tensix_x> <tensix_y> <config_l1_addr> [nbytes] [pcie_x] [pcie_y]
//   defaults: nbytes=1024, pcie=(2,0)
//
// RISK: a malformed write to the PCIe tile can hang the NoC, which would require
// tt-smi -r and kill the running X280 Linux. We mirror the proven Tensix write
// exactly (same dest tile, same host addr on the wire) to stay on the safe path.

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

// Blackhole PCIe transactions require bit 60 of the NoC local address to be set
// so the PCIe tile treats it as a DIRECT (non-iATU) 64-bit host address rather
// than an iATU-translated one (see noc_parameters.h NOC_XY_PCIE_ENCODING /
// NOC_LOCAL_ADDR mask 0x1000000FFFFFFFFF). Without it, host writes/reads hit a
// missing iATU entry and read back 0xffffffff. Bit 60 lands in window reg[1].
#define PCIE_NONIATU_BIT (1ULL << 60)

// D2H sender config-buffer word offsets. L1 alignment is 16 B on Blackhole, so
// sender_socket_md (32 B) + bytes_acked[1] (16 B) + sender_downstream_encoding
// (16 B) places the encoding at word 12. See tt_metal/hw/inc/hostdev/socket.h
// and D2HSocket::write_socket_metadata.
#define W_BSENT_LO 3   // downstream_bytes_sent_addr (host bytes_sent, low 32)
#define W_FIFO_LO 4    // downstream_fifo_addr       (host data FIFO,  low 32)
#define W_FIFO_SZ 5    // downstream_fifo_total_size
#define W_IS_D2H 6     // is_d2h
#define W_BSENT_HI 12  // d2h.bytes_sent_addr_hi
#define W_DATA_HI 13   // d2h.data_addr_hi
#define W_PCIE_ENC 14  // d2h.pcie_xy_enc (translated coord, encoded x|(y<<6))

static volatile uint32_t* g_cfg;  // TLB config register page

static void program_window(int win, uint64_t noc_addr, unsigned x, unsigned y) {
    volatile uint32_t* reg = g_cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(noc_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(noc_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    reg[3] = 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tensix_x> <tensix_y> <config_l1_addr> [nbytes] [pcie_x] [pcie_y]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    uint32_t nbytes = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 1024;
    unsigned px = argc > 5 ? atoi(argv[5]) : 2;  // PCIe tile physical NOC0 x
    unsigned py = argc > 6 ? atoi(argv[6]) : 0;  // PCIe tile physical NOC0 y

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    g_cfg = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE);
    if (g_cfg == MAP_FAILED) {
        perror("mmap cfg");
        return 1;
    }

    // --- (1) read the socket config buffer from the Tensix sender core L1 ---
    program_window(0, cfg_addr, tx, ty);
    volatile uint8_t* win0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE + 0 * WINDOW_2M_SIZE);
    if (win0 == MAP_FAILED) {
        perror("mmap win0");
        return 1;
    }
    volatile uint32_t* cfg = (volatile uint32_t*)(win0 + (cfg_addr & WINDOW_2M_MASK));
    uint32_t w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = cfg[i];
    }
    uint64_t host_data = ((uint64_t)w[W_DATA_HI] << 32) | w[W_FIFO_LO];
    uint64_t host_bsent = ((uint64_t)w[W_BSENT_HI] << 32) | w[W_BSENT_LO];
    uint32_t pcie_enc = w[W_PCIE_ENC];
    uint32_t fifo_sz = w[W_FIFO_SZ];

    printf("read config from Tensix (%u,%u):0x%lx ->\n", tx, ty, cfg_addr);
    printf("  host data FIFO addr  : 0x%016lx\n", host_data);
    printf("  host bytes_sent addr : 0x%016lx\n", host_bsent);
    printf(
        "  pcie_xy_enc          : 0x%08x  (translated x=%u y=%u)\n", pcie_enc, pcie_enc & 0x3f, (pcie_enc >> 6) & 0x3f);
    printf("  fifo_total_size      : %u   is_d2h=%u\n", fifo_sz, w[W_IS_D2H]);

    if (!w[W_IS_D2H] || host_data == 0) {
        fprintf(
            stderr, "config does not look like a live D2H socket (is_d2h=%u, data=0x%lx)\n", w[W_IS_D2H], host_data);
        return 2;
    }
    if (nbytes > fifo_sz) {
        fprintf(stderr, "nbytes %u > fifo_total_size %u\n", nbytes, fifo_sz);
        return 2;
    }
    if ((host_data >> WINDOW_2M_SHIFT) != (host_bsent >> WINDOW_2M_SHIFT)) {
        fprintf(
            stderr,
            "data and bytes_sent are in different 2MB pages (0x%lx vs 0x%lx); "
            "re-run (host IOVA varies) or extend tool for two windows\n",
            host_data,
            host_bsent);
        return 2;
    }

    // --- (2) write the data pattern to the host FIFO through the PCIe tile ---
    // bit 60 selects the PCIe tile's direct 64-bit host path (non-iATU).
    program_window(1, host_data | PCIE_NONIATU_BIT, px, py);
    volatile uint8_t* win1 =
        mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + 1 * WINDOW_2M_SIZE);
    if (win1 == MAP_FAILED) {
        perror("mmap win1");
        return 1;
    }
    volatile uint32_t* dp = (volatile uint32_t*)(win1 + (host_data & WINDOW_2M_MASK));
    uint32_t nwords = nbytes / 4;

    // Pre-write read probe: a NoC read to the PCIe tile is lower-risk than a
    // write for the first touch of a new tile/path. The host pinned buffer was
    // zero-initialized, so this should return 0 (and, crucially, return at all).
    volatile uint32_t probe = dp[0];
    printf("pre-write read probe: host[0] via PCIe tile (%u,%u) = 0x%08x (expect 0)\n", px, py, probe);

    printf("writing %u B (pattern 0xD2xxxxxx) to PCIe tile (%u,%u) -> host 0x%lx ...\n", nbytes, px, py, host_data);
    for (uint32_t i = 0; i < nwords; i++) {
        dp[i] = 0xD2000000u | i;  // recognizable: host should see 0xD2000000, 0xD2000001, ...
    }

    // --- (3) fence, then bump bytes_sent so the host read() releases the page ---
    // Read-back through the same window forces the posted writes to drain (a NoC
    // read is non-posted and ordered after the prior writes to the same tile).
    volatile uint32_t rb = dp[nwords - 1];
    volatile uint32_t* bsp = (volatile uint32_t*)(win1 + (host_bsent & WINDOW_2M_MASK));
    *bsp = nbytes;  // cumulative bytes_sent; first/only write, bytes_acked starts at 0
    volatile uint32_t rb2 = *bsp;
    printf("wrote bytes_sent=%u (readback data[last]=0x%08x bytes_sent=0x%08x); done.\n", nbytes, rb, rb2);
    return 0;
}

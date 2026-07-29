// X280 -> HOST D2H write, WRITE-ONLY (bh-14 safe path).
//
// Why a rewrite of x280_d2h_write.c: on bh-14 a NoC READ issued by the X280 into
// the PCIe tile / host-memory path that does not return a completion STALLS the
// in-order hart forever (it wedged the guest during the read-scan, 2026-06-18).
// So this tool issues NO reads through the PCIe tile -- not the pre-write probe,
// not the read-back fence. It only:
//   1. READS the D2H socket config from the Tensix sender core L1 (a plain Tensix
//      NoC read -- proven safe, what the pollers do) to learn the per-run host
//      addrs + pcie encoding.
//   2. WRITES the data pages to the host FIFO through the PCIe tile (posted).
//   3. Issues a CPU fence, then WRITES bytes_sent (posted). PCIe posted writes to
//      the same function keep order, so the host sees data before the release.
//
// Addressing comes straight from metal's PROVEN selftest path:
//   - PCIe tile coord  = the TRANSLATED coord metal uses, which is exactly the
//     socket's pcie_xy_enc (x | (y<<6)). The X280 small-window props_lo register
//     uses the same x|(y<<6) layout, so reg[2] = pcie_enc verbatim. (0x613 = the
//     translated PCIe tile (19,24) on bh-14 chip 0.)
//   - winsel 0 (NoC addr bits[63:58]=0): the host IOVA is a plain direct-IOMMU
//     address; NO bit-60 / non-iATU games (those were a bh-qb-05 misdirection).
//
// usage (run as root on the X280, needs /dev/mem):
//   ./x280_d2h_send <tensix_x> <tensix_y> <config_l1_addr> [nbytes]
//   defaults: nbytes=1024.  e.g. on bh-14: ./x280_d2h_send 1 2 0x17ffc0 1024

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

// D2H sender config-buffer word offsets (see tt_metal/hw/inc/hostdev/socket.h
// and D2HSocket::write_socket_metadata; L1 align 16 B on Blackhole).
#define W_BSENT_LO 3   // downstream_bytes_sent_addr (host bytes_sent, low 32)
#define W_FIFO_LO 4    // downstream_fifo_addr       (host data FIFO,  low 32)
#define W_FIFO_SZ 5    // downstream_fifo_total_size
#define W_IS_D2H 6     // is_d2h
#define W_BSENT_HI 12  // d2h.bytes_sent_addr_hi
#define W_DATA_HI 13   // d2h.data_addr_hi
#define W_PCIE_ENC 14  // d2h.pcie_xy_enc (translated x|(y<<6))

static volatile uint32_t* g_cfg;  // TLB config register page

// Program a small (2MB) TLB window: target NoC addr + (x,y) already encoded as
// props_lo low 12 bits. winsel/bit60 are folded into `noc_addr` by the caller.
static void program_window(int win, uint64_t noc_addr, uint32_t xy_enc) {
    volatile uint32_t* reg = g_cfg + (win * 0x10) / 4;
    reg[0] = (uint32_t)(noc_addr >> WINDOW_2M_SHIFT);
    reg[1] = (uint32_t)(noc_addr >> (WINDOW_2M_SHIFT + 32));
    reg[2] = xy_enc & 0xFFF;  // x_end[5:0] | y_end[11:6]; pcie_enc is exactly this
    reg[3] = 0;               // default ordering, NoC0
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tensix_x> <tensix_y> <config_l1_addr> [nbytes]\n", argv[0]);
        return 1;
    }
    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t cfg_addr = strtoull(argv[3], 0, 0);
    uint32_t nbytes = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 1024;

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

    // --- (1) read the socket config from the Tensix sender core L1 (safe) ---
    program_window(0, cfg_addr, (tx & 0x3f) | ((ty & 0x3f) << 6));
    volatile uint8_t* win0 = mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE);
    if (win0 == MAP_FAILED) {
        perror("mmap win0");
        return 1;
    }
    volatile uint32_t* cfg = (volatile uint32_t*)(win0 + (cfg_addr & WINDOW_2M_MASK));
    uint32_t w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = cfg[i];
    }
    munmap((void*)win0, WINDOW_2M_SIZE);

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
    fflush(stdout);  // make sure this lands even if a later store wedges things

    if (!w[W_IS_D2H] || host_data == 0) {
        fprintf(stderr, "config does not look like a live D2H socket\n");
        return 2;
    }
    if (nbytes > fifo_sz) {
        fprintf(stderr, "nbytes %u > fifo_total_size %u\n", nbytes, fifo_sz);
        return 2;
    }
    if ((host_data >> WINDOW_2M_SHIFT) != (host_bsent >> WINDOW_2M_SHIFT)) {
        fprintf(
            stderr, "data 0x%lx and bytes_sent 0x%lx span different 2MB pages; extend tool\n", host_data, host_bsent);
        return 2;
    }

    // --- (2) WRITE the data pages to the host FIFO through the PCIe tile ---
    // winsel 0, no bit60: target == plain host IOVA. Coord == pcie_enc verbatim.
    program_window(1, host_data, pcie_enc);
    volatile uint8_t* win1 =
        mmap(0, WINDOW_2M_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WINDOW_2M_BASE + 1 * WINDOW_2M_SIZE);
    if (win1 == MAP_FAILED) {
        perror("mmap win1");
        return 1;
    }
    volatile uint32_t* dp = (volatile uint32_t*)(win1 + (host_data & WINDOW_2M_MASK));
    uint32_t nwords = nbytes / 4;

    printf(
        "writing %u B (pattern 0xD2xxxxxx) to PCIe tile enc 0x%x -> host 0x%lx ... (write-only)\n",
        nbytes,
        pcie_enc,
        host_data);
    fflush(stdout);
    for (uint32_t i = 0; i < nwords; i++) {
        dp[i] = 0xD2000000u | i;  // host should see 0xD2000000, 0xD2000001, ...
    }

    // --- (3) fence (order our stores), then release via bytes_sent (posted) ---
    // No NoC read-back fence: PCIe posted writes to the same function are ordered,
    // so data lands before the bytes_sent release the host spins on.
    __sync_synchronize();
    volatile uint32_t* bsp = (volatile uint32_t*)(win1 + (host_bsent & WINDOW_2M_MASK));
    *bsp = nbytes;  // cumulative bytes_sent; first/only write, bytes_acked starts 0
    __sync_synchronize();
    printf("wrote bytes_sent=%u; done (no PCIe-tile reads issued).\n", nbytes);
    return 0;
}

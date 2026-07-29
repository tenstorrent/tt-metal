// Drive the X280's OWN NoC0 NIU as a master to issue ONE async NoC read of a
// remote Tensix core's L1, landing the response directly in X280 DRAM. This is
// the cmd-buffer path (real DMA engine), not the hardware-TLB-window path the
// pollers use — it decouples the in-order X280 from per-flit round-trip latency.
//
// Mechanism (from tt-metal blackhole/noc_nonblocking_api.h ncrisc_noc_fast_read
// + noc_init, and tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md):
//   - Program cmd buf 0 of NIU0 (X280 local aperture base 0x20056000):
//       TARG = remote Tensix (x,y):l1_addr   (the thing we read)
//       RET  = our OWN NoC coordinate : X280-physical dest   (where it lands)
//       CTRL = CPY|RD|RESP_MARKED|VC_STATIC|STATIC_VC(1);  AT_LEN_BE = nbytes
//       CMD_CTRL bit0 = SEND_REQ  -> fires.
//   - An inbound NoC write addressed to our own L2CPU tile passes through to the
//     X280 physical address space (lower 47 bits), so RET local offset == the
//     X280 physical DRAM address. DRAM uncached base = 0x30000000.
//   - Completion: NIU_MST_REQS_OUTSTANDING_ID(0) returns to 0 (and
//     RD_RESP_RECEIVED increments). We poll with a BOUNDED loop — a malformed
//     transaction can wedge the NoC, and recovery needs tt-smi -r which kills
//     X280 Linux, so we never spin forever and never reset.
//
// Run as root (needs /dev/mem + pagemap PFNs).
//   ./niu_read                              # SAFE probe: print our own NoC coord
//   ./niu_read <tx> <ty> <l1_addr> [nbytes] # fire ONE read (default 64B)
//
// <tx> <ty> are the remote Tensix NOC0 coords printed by the host launcher.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define NIU0_BASE 0x20056000UL  // X280-local aperture for NoC0 NIU registers

// cmd-buf 0 register byte offsets within the NIU aperture (== NOC_REGS_START_ADDR layout)
#define NOC_TARG_ADDR_LO 0x00
#define NOC_TARG_ADDR_MID 0x04
#define NOC_TARG_ADDR_COORD 0x08  // = NOC_TARG_ADDR_HI
#define NOC_RET_ADDR_LO 0x0C
#define NOC_RET_ADDR_MID 0x10
#define NOC_RET_ADDR_COORD 0x14  // = NOC_RET_ADDR_HI
#define NOC_PACKET_TAG 0x18
#define NOC_CTRL 0x1C
#define NOC_AT_LEN_BE 0x20
#define NOC_CMD_CTRL 0x40
#define NOC_NODE_ID 0x44
#define NOC_CFG_BASE 0x100   // NOC_CFG(cnt) = base + cnt*4
#define NOC_ID_LOGICAL 0x12  // NOC_CFG index; format {logical_y[5:0], logical_x[5:0]}
#define NOC_STATUS_BASE 0x200

// NOC_STATUS counter indices
#define ST_RD_RESP_RECEIVED 0x2
#define ST_CMD_ACCEPTED 0x4
#define ST_RD_REQ_SENT 0x5
#define ST_REQS_OUTSTANDING_0 0x10

// NOC_CTRL command fields (blackhole noc_parameters.h)
#define NOC_CMD_CPY (0x0 << 0)
#define NOC_CMD_RD (0x0 << 1)
#define NOC_CMD_RESP_MARKED (0x1 << 4)
#define NOC_CMD_VC_STATIC (0x1 << 7)
#define NOC_CMD_STATIC_VC(vc) (((uint32_t)(vc)) << 13)
#define NOC_CTRL_SEND_REQ (0x1 << 0)
#define NOC_ADDR_COORD_SHIFT 36

// coordinate word as written to *_ADDR_COORDINATE: {y[5:0], x[5:0]}
#define XY_COORD(x, y) ((((uint32_t)(y)) << 6) | ((uint32_t)(x)))

static volatile uint32_t* niu;  // mmap'd NIU0 register page (RW)
static uint32_t rd(uint32_t off) { return niu[off / 4]; }
static void wr(uint32_t off, uint32_t v) { niu[off / 4] = v; }
static uint32_t ctr(int cnt) { return niu[(NOC_STATUS_BASE + cnt * 4) / 4]; }

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

// Translate a locked virtual address to its physical address via pagemap (root).
static uint64_t virt_to_phys(int pmfd, void* va) {
    uint64_t vaddr = (uint64_t)va;
    uint64_t entry = 0;
    if (pread(pmfd, &entry, sizeof entry, (vaddr / 4096) * 8) != sizeof entry) {
        return 0;
    }
    if (!(entry & (1ULL << 63))) {
        return 0;  // not present
    }
    return ((entry & ((1ULL << 55) - 1)) << 12) | (vaddr & 0xfff);
}

int main(int argc, char** argv) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    niu = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, NIU0_BASE);
    if (niu == MAP_FAILED) {
        perror("mmap niu");
        return 1;
    }

    // Discover our own NoC coordinate (the read response's return coordinate).
    uint32_t node_id = rd(NOC_NODE_ID);
    uint32_t logical = rd(NOC_CFG_BASE + NOC_ID_LOGICAL * 4);
    uint32_t phys_x = node_id & 0x3f, phys_y = (node_id >> 6) & 0x3f;
    uint32_t log_x = logical & 0x3f, log_y = (logical >> 6) & 0x3f;
    printf("X280 NoC0 NIU @ 0x%lx\n", NIU0_BASE);
    printf("  NOC_NODE_ID    = 0x%08x -> phys    (x=%u, y=%u)\n", node_id, phys_x, phys_y);
    printf("  NOC_ID_LOGICAL = 0x%08x -> logical (x=%u, y=%u)\n", logical, log_x, log_y);

    if (argc < 4) {
        printf("\n[probe only] pass <tx> <ty> <l1_addr> [nbytes] to fire one read.\n");
        return 0;
    }

    unsigned tx = atoi(argv[1]), ty = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    uint32_t nbytes = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 64;

    // --- landing buffer in X280 DRAM ----------------------------------------
    // One locked page; sentinel-fill so we can see the NoC overwrite it. Read it
    // back through an UNCACHED /dev/mem alias at the same physical address so the
    // X280 dcache can't hand us a stale sentinel.
    size_t pgsz = 4096;
    void* buf = mmap(0, pgsz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (buf == MAP_FAILED) {
        perror("mmap landing buf");
        return 1;
    }
    memset(buf, 0xA5, pgsz);  // sentinel
    int pmfd = open("/proc/self/pagemap", O_RDONLY);
    if (pmfd < 0) {
        perror("open pagemap");
        return 1;
    }
    uint64_t phys = virt_to_phys(pmfd, buf);
    if (!phys) {
        fprintf(stderr, "pagemap gave no PFN (run as root, kernel must expose PFNs)\n");
        return 1;
    }
    printf("\nlanding buf: va=%p phys=0x%lx (X280 DRAM uncached base is 0x30000000)\n", buf, phys);
    if (phys < 0x30000000UL) {
        fprintf(
            stderr,
            "WARNING: phys 0x%lx is below the documented DRAM NoC base 0x30000000.\n"
            "Linux-physical may differ from the NoC passthrough address; not firing.\n"
            "Re-run after confirming the DRAM physical base.\n",
            phys);
        return 1;
    }
    volatile uint8_t* uview = mmap(0, pgsz, PROT_READ, MAP_SHARED, fd, phys & ~(pgsz - 1));
    if (uview == MAP_FAILED) {
        perror("mmap uncached view (/dev/mem on RAM may be blocked by STRICT_DEVMEM)");
        uview = 0;  // fall back to cached buf read + a warning
    }

    // --- program cmd buf 0 and fire ONE read --------------------------------
    uint32_t want_resp = ctr(ST_RD_RESP_RECEIVED) + 1;
    uint32_t b_req = ctr(ST_RD_REQ_SENT), b_acc = ctr(ST_CMD_ACCEPTED), b_out = ctr(ST_REQS_OUTSTANDING_0);

    wr(NOC_CTRL, NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1));
    wr(NOC_PACKET_TAG, 0);  // transaction id 0 -> tracked in REQS_OUTSTANDING_ID(0)
    // TARG = remote Tensix L1
    wr(NOC_TARG_ADDR_LO, (uint32_t)l1_addr);
    wr(NOC_TARG_ADDR_MID, (uint32_t)(l1_addr >> 32) & 0xF);
    wr(NOC_TARG_ADDR_COORD, XY_COORD(tx, ty));
    // RET = our own coordinate : X280 physical dest (passthrough to DRAM).
    // Use the LOGICAL/translated coord (as noc_init does) — same space as the
    // Tensix TARG coords the host launcher prints — not the raw NODE_ID.
    wr(NOC_RET_ADDR_LO, (uint32_t)phys);
    wr(NOC_RET_ADDR_MID, (uint32_t)(phys >> 32) & 0xF);
    wr(NOC_RET_ADDR_COORD, XY_COORD(log_x, log_y));
    wr(NOC_AT_LEN_BE, nbytes);

    double t0 = now_ns();
    wr(NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);  // GO

    // Bounded completion poll — never spin forever (hang -> would need tt-smi -r).
    const long MAX_SPINS = 50000000L;
    long spins = 0;
    while (ctr(ST_REQS_OUTSTANDING_0) != b_out && spins < MAX_SPINS) {
        spins++;
    }
    while (ctr(ST_RD_RESP_RECEIVED) != want_resp && spins < MAX_SPINS) {
        spins++;
    }
    double ns = now_ns() - t0;

    uint32_t a_req = ctr(ST_RD_REQ_SENT), a_acc = ctr(ST_CMD_ACCEPTED), a_resp = ctr(ST_RD_RESP_RECEIVED),
             a_out = ctr(ST_REQS_OUTSTANDING_0);
    int completed = (a_resp == want_resp) && (a_out == b_out);

    printf("\nfired 1 read: %u B from Tensix (%u,%u):0x%lx -> phys 0x%lx\n", nbytes, tx, ty, l1_addr, phys);
    printf("  %s in %.0f ns (%ld spins)\n", completed ? "COMPLETED" : "DID NOT COMPLETE", ns, spins);
    printf("  RD_REQ_SENT      %+d\n", (int)(a_req - b_req));
    printf("  CMD_ACCEPTED     %+d\n", (int)(a_acc - b_acc));
    printf("  RD_RESP_RECEIVED %+d\n", (int)(a_resp - (want_resp - 1)));
    printf("  REQS_OUTSTAND(0) before=%u after=%u\n", b_out, a_out);
    if (!completed) {
        fprintf(stderr, "  WARNING: not confirmed complete — check coords/addr before retrying.\n");
    }

    // Show the landed bytes (uncached alias if we have it, else cached buffer).
    const volatile uint8_t* src = uview ? uview + (phys & (pgsz - 1)) : (const volatile uint8_t*)buf;
    printf("  landed[0..%u] (%s):", nbytes < 32 ? nbytes : 32, uview ? "uncached" : "CACHED-may-be-stale");
    for (uint32_t i = 0; i < nbytes && i < 32; i++) {
        printf(" %02x", src[i]);
    }
    printf("\n  (sentinel was 0xA5; first u32 = 0x%08x)\n", *(const volatile uint32_t*)src);
    return 0;
}

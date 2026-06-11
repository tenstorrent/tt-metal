// READ-ONLY probe of the X280's own NoC0 NIU register block.
//
// Hypothesis: the L2CPU "NIU #0 config/status" aperture at 0x20056000 is the
// X280-local view of the standard NIU register base (0xFFB20000 on Tensix), so
// NoC status counters live at 0x20056000 + 0x200 + cnt*4 (NOC_STATUS(cnt)).
//
// Confirm it safely: snapshot the read counters, do a known number of TLB-window
// reads (which flow through THIS NIU), snapshot again. If RD_REQ_SENT /
// RD_RESP_RECEIVED advance by ~that many, the mapping is confirmed and we know
// exactly where the command buffers are (base + i*0x800).
//
// Run as root: ./niu_probe <x> <y> <l1_addr> <nreads>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define NIU0_BASE 0x20056000UL  // X280 local aperture for NoC0 NIU config/status
#define NOC_STATUS_OFF 0x200    // NOC_STATUS(cnt) = base + 0x200 + cnt*4

// counter indices (from tt-metal noc_parameters.h)
#define RD_RESP_RECEIVED 0x2
#define RD_DATA_WORD_RECEIVED 0x3
#define CMD_ACCEPTED 0x4
#define RD_REQ_SENT 0x5
#define RD_REQ_STARTED 0xE
#define REQS_OUTSTANDING_0 0x10
#define REQS_OUTSTANDING_1 0x11

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)

static volatile uint32_t* niu;
static uint32_t ctr(int cnt) { return niu[(NOC_STATUS_OFF + cnt * 4) / 4]; }

int main(int argc, char** argv) {
    unsigned x = atoi(argv[1]), y = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    long nreads = argc > 4 ? atol(argv[4]) : 100000;
    int win = 200;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    // Map the NIU0 config/status page (read-only).
    niu = mmap(0, 4096, PROT_READ, MAP_SHARED, fd, NIU0_BASE);
    if (niu == MAP_FAILED) {
        perror("mmap niu");
        return 1;
    }

    // A TLB window to drive read traffic through the NIU.
    volatile uint32_t* cfg =
        mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE + win * 0x10 & ~0xfffUL);
    volatile uint32_t* reg = cfg + ((win * 0x10) & 0xfff) / 4;
    reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    reg[3] = 0;
    volatile uint32_t* w =
        mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE + (uint64_t)win * WINDOW_2M_SIZE);
    volatile uint32_t* flit = w + (l1_addr & (WINDOW_2M_SIZE - 1)) / 4;

    printf("NIU0 @ 0x%lx  raw words 0x200..0x21c:", NIU0_BASE);
    for (int o = 0x200; o <= 0x21c; o += 4) {
        printf(" %08x", niu[o / 4]);
    }
    printf("\n");

    uint32_t a_req = ctr(RD_REQ_SENT), a_resp = ctr(RD_RESP_RECEIVED), a_word = ctr(RD_DATA_WORD_RECEIVED),
             a_acc = ctr(CMD_ACCEPTED), a_out0 = ctr(REQS_OUTSTANDING_0);

    volatile uint32_t sink = 0;
    for (long i = 0; i < nreads; i++) {
        sink ^= flit[0];  // nreads single-u32 reads
    }

    uint32_t b_req = ctr(RD_REQ_SENT), b_resp = ctr(RD_RESP_RECEIVED), b_word = ctr(RD_DATA_WORD_RECEIVED),
             b_acc = ctr(CMD_ACCEPTED), b_out0 = ctr(REQS_OUTSTANDING_0);

    printf("did %ld single-u32 reads of (%u,%u):0x%lx  (sink=%u)\n", nreads, x, y, l1_addr, sink);
    printf("counter          before        after        delta\n");
    printf("RD_REQ_SENT      %10u  %10u  %+d\n", a_req, b_req, (int)(b_req - a_req));
    printf("RD_RESP_RECEIVED %10u  %10u  %+d\n", a_resp, b_resp, (int)(b_resp - a_resp));
    printf("RD_DATA_WORD_RCV %10u  %10u  %+d\n", a_word, b_word, (int)(b_word - a_word));
    printf("CMD_ACCEPTED     %10u  %10u  %+d\n", a_acc, b_acc, (int)(b_acc - a_acc));
    printf("REQS_OUTSTAND(0) %10u  %10u  %+d\n", a_out0, b_out0, (int)(b_out0 - a_out0));
    return 0;
}

// Poll a Tensix L1 counter from the X280 via a 2MB NoC TLB window.
// Layout per tt-bh-linux docs/addressing.md. Run as root.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define TLB_2M_CONFIG_BASE 0x2ff00000UL
#define WINDOW_2M_BASE (0x30000000UL + 0x400000000UL)
#define WINDOW_2M_SHIFT 21
#define WINDOW_2M_SIZE (1UL << WINDOW_2M_SHIFT)

int main(int argc, char** argv) {
    unsigned x = atoi(argv[1]), y = atoi(argv[2]);
    uint64_t l1_addr = strtoull(argv[3], 0, 0);
    int win = 200;  // high index to avoid windows used by the kernel

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    volatile uint32_t* cfg =
        mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TLB_2M_CONFIG_BASE + win * 0x10 & ~0xfffUL);
    if (cfg == MAP_FAILED) {
        perror("mmap cfg");
        return 1;
    }
    volatile uint32_t* reg = cfg + ((win * 0x10) & 0xfff) / 4;
    reg[0] = (uint32_t)(l1_addr >> WINDOW_2M_SHIFT);
    reg[1] = 0;
    reg[2] = (x & 0x3f) | ((y & 0x3f) << 6);  // x_end, y_end: unicast
    reg[3] = 0;

    volatile uint32_t* w =
        mmap(0, WINDOW_2M_SIZE, PROT_READ, MAP_SHARED, fd, WINDOW_2M_BASE + (uint64_t)win * WINDOW_2M_SIZE);
    if (w == MAP_FAILED) {
        perror("mmap win");
        return 1;
    }
    volatile uint32_t* counter = w + (l1_addr & (WINDOW_2M_SIZE - 1)) / 4;

    for (;;) {
        printf("counter = %u\n", *counter);
        fflush(stdout);
        usleep(20000);
    }
}

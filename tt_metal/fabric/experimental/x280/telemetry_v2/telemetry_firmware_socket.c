#include <stdint.h>
#include <stddef.h>

#include "rt/x280_hw.h"
#include "noc.h"
#include "rt/x280_idle.h"

// TODO: Put in common place so both sides can grab it.
#define kConfigAddr 0x08133000



typedef struct {
    uint32_t bytes_sent;
    uint32_t num_downstreams;
    uint32_t write_ptr;
    uint32_t bytes_sent_addr_lo;
    uint32_t fifo_addr_lo;
    uint32_t fifo_size;
    uint32_t is_d2h;
    uint32_t pad0;
    uint32_t bytes_acked;
    uint32_t pad1[3];
    uint32_t bytes_sent_addr_hi;
    uint32_t fifo_addr_hi;
    uint32_t pcie_xy_enc;
    uint32_t pad2;
} D2HSocketConfig;

_Static_assert(offsetof(D2HSocketConfig, bytes_sent) == 0u, "bytes_sent");
_Static_assert(offsetof(D2HSocketConfig, is_d2h) == 24u, "is_d2h");
_Static_assert(offsetof(D2HSocketConfig, bytes_acked) == 32u, "bytes_acked");
_Static_assert(offsetof(D2HSocketConfig, bytes_sent_addr_hi) == 48u, "enc");
_Static_assert(sizeof(D2HSocketConfig) == 64u, "size");

static volatile D2HSocketConfig* const g_socket_config = (volatile D2HSocketConfig*) kConfigAddr;


int main(uint64_t hartid) {
    if (hartid == 0) {


    }

    for (;;) {
        x280_pause();
    }
}
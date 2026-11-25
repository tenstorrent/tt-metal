#include <stdint.h>
#include <stddef.h>

#include "rt/x280_hw.h"
#include "noc.h"
#include "rt/x280_idle.h"

#include "common/heart_beat.h"

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


static volatile Heartbeat* g_heartbeat = (volatile Heartbeat*)X280_STANDALONE_HEARTBEAT_ADDR;
const uint64_t g_ticks_per_ms = X280_MTIME_HZ / 1000;
const uint64_t g_ticks_per_2s = g_ticks_per_ms * 2000;
uint64_t g_next_heartbeat = 1867;

void heart_beat() {
    if (x280_mtime() >= g_next_heartbeat) {
        g_next_heartbeat += g_ticks_per_ms;
        g_heartbeat->counter = g_heartbeat->counter + 1;
        if (g_heartbeat->counter >= UINT64_MAX-1) {
            g_heartbeat->counter = 1;
        }
    }
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        g_heartbeat->counter = 0;
        g_heartbeat->fifo_addr_lo = g_socket_config->fifo_addr_lo;
        g_heartbeat->fifo_addr_hi = g_socket_config->fifo_addr_hi;
        g_heartbeat->fifo_size = g_socket_config->fifo_size;

        x280_fence_ow();
        g_heartbeat->magic = X280_STANDALONE_HEARTBEAT_MAGIC;  

        uint64_t next_work = x280_mtime() + g_ticks_per_2s;
        for(;;) {

            if (x280_mtime() >= next_work) {
                next_work += g_ticks_per_2s;
                // result = read_arc();
                // if (result) {
                //     update_arc_data_in_table();
                // }       
            }

            heart_beat();
            x280_pause();
        }
    }

    for (;;) {
        x280_pause();
    }
}
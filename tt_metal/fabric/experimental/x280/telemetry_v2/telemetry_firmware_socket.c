#include <stdint.h>
#include <stddef.h>

#include "rt/x280_hw.h"
#include "noc.h"
#include "rt/x280_idle.h"

#include "common/heart_beat.h"
#include "telemetry_helpers.h"

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

// static x280_noc_coord_t g_coords;
// static uint64_t g_fifo_addr;
// static uint64_t g_bytes_sent_addr;

void write_data_to_socket(
    const uint8_t* src_bytes, 
    size_t num_bytes,
    const x280_noc_coord_t* coords,
    uint64_t fifo_addr,
    uint64_t bytes_sent_addr
) {
    // Write the data.
    volatile uint8_t* fifo_ptr = NULL;
    noc_window_status_t win_status = noc_open_window_2m(
        16,             
        *coords,
        fifo_addr,    /* The address where the fifo lives. */
        NOC_ORDER_STRICT,
        (volatile void**)&fifo_ptr);
    if (win_status != NOC_WINDOW_OPEN) {
        goto cleanup;
    }

    copy_memory(fifo_ptr, src_bytes, num_bytes);

    // 0 Pad any left-over data.
    uint32_t total_bytes = num_bytes;
    uint32_t pad_bytes = (64 - num_bytes % 64) % 64;
    for(int i = num_bytes; i < num_bytes + pad_bytes; ++i) {
        *(fifo_ptr + i) = 0;
        ++total_bytes;
    }

    x280_fence_iorw();

    // Now let the other side know how much data we wrote.
    volatile uint32_t* bytes_sent_ptr = NULL;
    win_status = noc_open_window_2m(
        16,             
        *coords,
        bytes_sent_addr,    /* The address where the fifo lives. */
        NOC_ORDER_STRICT,
        (volatile void**)&bytes_sent_ptr);
    if (win_status != NOC_WINDOW_OPEN) {
        goto cleanup;
    }

    *bytes_sent_ptr = total_bytes;
    
cleanup:
    noc_clear_tlb_2m(16);
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        clear_memory(g_heartbeat, sizeof(Heartbeat));
        g_heartbeat->counter = 0;
        x280_fence_ow();
        g_heartbeat->magic = X280_STANDALONE_HEARTBEAT_MAGIC;  

        // Make sure config is there.
        // TODO: Should we give up at some point.
        for(;;) {
            x280_fence_iorw();
            if (g_socket_config->is_d2h == 1 &&
                g_socket_config->fifo_size >= 1) {
                break;
            } 
            x280_pause();
        }

        // Fifo Address.
        uint64_t addr_high = g_socket_config->fifo_addr_hi;
        uint64_t addr_lo =  g_socket_config->fifo_addr_lo;
        uint64_t fifo_addr = addr_high << 32;
        fifo_addr += addr_lo;

        // Bytes sent addr.
        addr_high = g_socket_config->bytes_sent_addr_hi;
        addr_lo = g_socket_config->bytes_sent_addr_lo;
        uint64_t bytes_sent_addr = addr_high << 32;
        bytes_sent_addr += addr_lo;

        // PCIE tile co-ords that we need to send data to host.
        uint32_t pcie_xy_enc = g_socket_config->pcie_xy_enc;
        x280_noc_coord_t coords = {
            .x = pcie_xy_enc & 0x3F,
            .y = (pcie_xy_enc >> 6) & 0x3F
        };

        // Send a message.
        const char* str = "Hello World";
        write_data_to_socket(
            (const uint8_t*) str, 
            str_len(str) + 1,
            &coords,
            fifo_addr,
            bytes_sent_addr
        );

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
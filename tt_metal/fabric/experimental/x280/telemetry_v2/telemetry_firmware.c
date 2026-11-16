// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone skeleton: booted straight from reset, no resident idle. Hart 0 publishes a
// "I'm alive" word and a counter in LIM; harts 1-3 park. It never returns.
// The heartbeat address and magic are shared with the host in x280_address_map.h.

#include <stdint.h>

#include "rt/x280_hw.h"
#include "arc_info.h"
#include "common/telemetry_table_v2.h"


static ArcInfo g_arc_info = {.do_not_put_into_bss_pal = 1867, .version = 0, .length = 0};

typedef struct {
    uint64_t magic;    // X280_STANDALONE_HEARTBEAT_MAGIC once the firmware is up; the host checks this first.
    uint64_t counter;  // Incremented about once a millisecond, so a changing value means it's running.
} Heartbeat;


bool read_arc(uint32_t run_value) {
    bool result = true;

    x280_noc_coord_t coords = {.x = 8, .y = 0};
    volatile uint32_t* arc_table_ptr = NULL;
    noc_window_status_t win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        0x80030434, /* SCRATCH_RAM_13 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_table_ptr);
    if (win_status != NOC_WINDOW_OPEN) {
        result = false;
        goto cleanup;
    }

    // Just keep looping untl we see the value goto non-zero.
    while (*arc_table_ptr == 0) {
        if (x280_hart_stop_requested(0, run_value)) {
            result = false;
            goto cleanup;
        }
    }

    // We now have the address of the table, map that in.
    volatile uint32_t* arc_table = NULL;
    win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        *arc_table_ptr, /* value in SCRATCH_RAM_13 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_table);
    if (win_status != NOC_WINDOW_OPEN) {
        result = false;
        goto cleanup;
    }

    g_arc_info.version = arc_table[0];
    g_arc_info.length = arc_table[1] > 256 ? 256 : arc_table[1];
    copy_u32_memory(g_arc_info.data, arc_table + 2, g_arc_info.length);

cleanup:
    noc_clear_tlb_2m(16);

    return result;
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        volatile Heartbeat* heartbeat = (volatile Heartbeat*)X280_STANDALONE_HEARTBEAT_ADDR;

        heartbeat->counter = 0;
        x280_fence_ow();
        heartbeat->magic = X280_STANDALONE_HEARTBEAT_MAGIC;  // Last, so the host never sees the magic with a stale counter.

        const uint64_t ticks_per_ms = X280_MTIME_HZ / 1000;
        uint64_t next = x280_mtime() + ticks_per_ms;
        const uint64_t ticks_per_2s = ticks_per_ms * 2000;
        uint64_t next_work = x280_mtime() + ticks_per_2s;
        for (;;) {
            if (x280_mtime() >= next) {
                next += ticks_per_ms;
                heartbeat->counter = heartbeat->counter + 1;
                if (heartbeat->counter >= UINT32_MAX-1) {
                    heartbeat->counter = 1;
                }
            }

            if (x280_mtime() >= next_work) {
                next_work += ticks_per_2s;
                //auto result = read_arc(run_value);
                //if (result) {
                    //update_arc_data_in_table();
                //}       
            }

            x280_pause();
        }
    }

    for (;;) {
        x280_pause();
    }
}

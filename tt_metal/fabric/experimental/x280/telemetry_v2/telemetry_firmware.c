// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone skeleton: booted straight from reset, no resident idle. Hart 0 publishes a
// "I'm alive" word and a counter in LIM; harts 1-3 park. It never returns.
// The heartbeat address and magic are shared with the host in x280_address_map.h.

#include <stdint.h>

#include "rt/x280_hw.h"

typedef struct {
    uint64_t magic;    // X280_STANDALONE_HEARTBEAT_MAGIC once the firmware is up; the host checks this first.
    uint64_t counter;  // Incremented about once a millisecond, so a changing value means it's running.
} Heartbeat;

int main(uint64_t hartid) {
    if (hartid == 0) {
        volatile Heartbeat* heartbeat = (volatile Heartbeat*)X280_STANDALONE_HEARTBEAT_ADDR;

        heartbeat->counter = 0;
        x280_fence_ow();
        heartbeat->magic = X280_STANDALONE_HEARTBEAT_MAGIC;  // Last, so the host never sees the magic with a stale counter.

        const uint64_t ticks_per_ms = X280_MTIME_HZ / 1000;
        uint64_t next = x280_mtime() + ticks_per_ms;
        for (;;) {
            if (x280_mtime() >= next) {
                next += ticks_per_ms;
                heartbeat->counter = heartbeat->counter + 1;
            }
            x280_pause();
        }
    }

    for (;;) {
        x280_pause();
    }
}

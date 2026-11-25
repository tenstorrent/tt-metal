// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "risc_common.h"

// Emits ZONE_COUNT profiler zones across a wrap of the wall clock's low word (every 2^32 cycles): wait until the low
// word is within WRAP_LEAD cycles of the wrap, then emit the zones back to back. A torn high/low read puts a marker
// 2^32 cycles early, which the host rejects as an orphan or mismatched ZONE_END.
void kernel_main() {
    constexpr uint32_t wrap_threshold = 0u - WRAP_LEAD;
    while (get_timestamp_32b() >= wrap_threshold) {
    }  // Already inside the lead window: let this wrap pass first.
    while (get_timestamp_32b() < wrap_threshold) {
    }
    for (uint32_t i = 0; i < ZONE_COUNT; i++) {
        DeviceZoneScopedN("WALL-CLOCK-WRAP");
        asm("nop");
    }
}

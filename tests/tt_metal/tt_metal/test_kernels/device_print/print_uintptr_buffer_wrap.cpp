// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

// Fill the per-core DPRINT ring buffer enough times to exercise wrap-around and aligned wpos
// advancement, then emit mixed-type sentinel prints.
void kernel_main() {
    constexpr uint32_t fill_count = 400;
    for (uint32_t i = 0; i < fill_count; i++) {
        DEVICE_PRINT("fill {}\n", i);
    }

    uint32_t idx = 1;
    float f32 = 2.5f;
    uint64_t u64 = 0x123456789ABCDEF0ull;
    DEVICE_PRINT("wrap idx={} float={:.1f} u64={}\n", idx, f32, u64);
    DEVICE_PRINT("wrap_done=1\n");
}

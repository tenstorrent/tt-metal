// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-FULL");
        DeviceTimestampedData("TEST", i + ((uint64_t)1 << 32));
        DeviceRecordEvent(i);
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < LOOP_SIZE; j++) {
            asm("nop");
        }
    }
}

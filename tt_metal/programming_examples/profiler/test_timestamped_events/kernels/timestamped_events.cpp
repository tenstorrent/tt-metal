// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-FULL");
        DeviceTimestampedData("TEST", i + ((uint64_t)1 << 32));
#if defined(ARCH_QUASAR)
        DeviceRecordEvent(i);  // still a distinct marker type on the DRAM backend; the pytest's
                               // 10-words-per-iteration buffer math depends on its 2-word size
#else
        DeviceTimestampedData("TEST-EVENT", i);  // the runtime-event type is gone on the streaming wire
#endif
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < LOOP_SIZE; j++) {
            asm("nop");
        }
    }
}

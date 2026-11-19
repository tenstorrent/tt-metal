// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// Two DeviceZoneScopedNIf zones per iteration, one active and one not (ZONE_CT_ON / ZONE_CT_OFF are kernel
// defines, so constant expressions). The inactive one must leave no marker and take no slot of the profiler
// buffer: LOOP_COUNT is chosen so the active zone fills most of it, so an inactive zone that still took slots
// would cut the active zone's count short (test_device_profiler.py).
void kernel_main() {
    constexpr bool ct_on = ZONE_CT_ON != 0;
    constexpr bool ct_off = ZONE_CT_OFF != 0;
    for (int i = 0; i < LOOP_COUNT; i++) {
        {
            DeviceZoneScopedNIf("TEST-CT-ON", ct_on);
            for (int j = 0; j < LOOP_SIZE; j++) {
                asm volatile("nop");
            }
        }
        {
            DeviceZoneScopedNIf("TEST-CT-OFF", ct_off);
            for (int j = 0; j < LOOP_SIZE; j++) {
                asm volatile("nop");
            }
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/debug/kernel_profiler.hpp"

// The compute-kernel twin of conditional_zones_dm.cpp: the same two zones on every TRISC.
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

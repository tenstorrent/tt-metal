// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute-side companion to profiler_zone_producer.cpp. One compute kernel is built for TRISC0/1/2, so
// this fills SPSC lanes 2, 3 and 4 -- the three a data-movement-only producer leaves empty. Together
// with a BRISC (lane 0) and an NCRISC (lane 1) producer, all five rings of a core are live at once,
// which is what actually exercises the drainer's per-lane indexing and its five-head write-back.
//
// Compile-time args rather than runtime, so no CB or runtime-arg plumbing is needed on the compute path.
//
// kernel_profiler.hpp must be included HERE, unlike on the data-movement path where it arrives through
// dataflow_api.h. trisck.cc includes chlkc_list.h (which pulls in this kernel) at line 13 and only then
// includes kernel_profiler.hpp at line 15 -- so by the time a compute kernel body is parsed the zone
// macros do not exist yet. Without this include the build fails with "'DeviceZoneScopedN' was not
// declared in this scope". It is also why no other compute kernel in the tree emits zones.

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kNumZones = get_compile_time_arg_val(0);
    constexpr uint32_t kWorkPerZone = get_compile_time_arg_val(1);

    for (uint32_t i = 0; i < kNumZones; i++) {
        DeviceZoneScopedN("E2E-PRODUCER-COMPUTE");
        for (volatile uint32_t s = 0; s < kWorkPerZone; s++) {
        }
    }
}

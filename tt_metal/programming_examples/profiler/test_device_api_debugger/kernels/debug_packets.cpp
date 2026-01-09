// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tools/profiler/event_metadata.hpp"
#include "tools/profiler/noc_event_profiler.hpp"

using EMD = KernelProfilerNocEventMetadata;

void kernel_main() {
#if defined(PROFILE_NOC_EVENTS)
    riscv_wait(START_DELAY);
    // Make this really large so it exceeds the max profiler sizes to test mid run dumps
    for (uint32_t i = 0; i < 10000; ++i) {
        noc_event_profiler::recordNocEvent(EMD::NocEventType::READ, my_x[0], my_y[0], 1000, 0, 0, i);
    }
#endif
}

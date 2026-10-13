// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One zone per metadata shape in debug_event_meta_silicon.h, for the silicon round trip in
// tests/tt_metal/tt_metal/debug_tools/debug_event/test_debug_event_meta_silicon.cpp. Built as a data-movement and as a
// compute kernel, so every RISC's image carries its own records.

#include <cstdint>

#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "api/compute/compute_kernel_api.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "api/debug/kernel_profiler.hpp"

#include "tests/tt_metal/tt_metal/test_kernels/misc/debug_event_meta_silicon.h"

namespace m = debug_event_meta_silicon;

template <uint32_t Value>
void templated_zone() {
    DeviceZoneScopedWith(m::TemplatedMeta{m::kTemplatedName, Value});
}

void kernel_main() {
    {
        DeviceZoneScopedN("meta_plain");
    }
    {
        DeviceZoneScopedNC("meta_colored", m::kColor);
    }
    {
        DeviceZoneScopedWith(m::kNested);
    }
    {
        DeviceZoneScopedWith(m::kDeep);
    }
    {
        DeviceZoneScopedWith(m::kAllWidths);
    }
    {
        DeviceZoneScopedWith(m::kDefaulted);
    }
    {
        DeviceZoneScopedWith(m::kTwoStrings);
    }
    templated_zone<m::kTemplatedA>();
    templated_zone<m::kTemplatedB>();
    templated_zone<m::kTemplatedA>();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One record per metadata shape in observability_event_meta_silicon.h, for the silicon round trip in
// tests/tt_metal/tt_metal/debug_tools/observability_event/test_observability_event_meta_silicon.cpp: the three zone
// macros, and a DevicePublish record without a payload for each metadata type of its own. Built as a data-movement and
// as a compute kernel, so every RISC's image carries its own call sites.

#include <cstdint>

#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "api/compute/compute_kernel_api.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "api/debug/device_publish.h"
#include "api/debug/kernel_profiler.hpp"

#include "tests/tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.h"

namespace m = observability_event_meta_silicon;

template <uint32_t Value>
void templated_record() {
    DevicePublish<m::TemplatedMeta{m::kTemplatedName, Value}>();
}

void kernel_main() {
    {
        DeviceZoneScopedN("meta_plain");
    }
    {
        DeviceZoneScopedNC("meta_colored", m::kColor);
    }
    {
        DeviceZoneScopedNCU("meta_unit", m::kColor, m::kUnit);
    }
    DevicePublish<m::kNested>();
    DevicePublish<m::kDeep>();
    DevicePublish<m::kAllWidths>();
    DevicePublish<m::kDefaulted>();
    DevicePublish<m::kTwoStrings>();
    templated_record<m::kTemplatedA>();
    templated_record<m::kTemplatedB>();
    templated_record<m::kTemplatedA>();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One zone per metadata shape in observability_event_meta_silicon.h, for the silicon round trip in
// tests/tt_metal/tt_metal/debug_tools/observability_event/test_observability_event_meta_silicon.cpp. Built as a
// data-movement and as a compute kernel, so every RISC's image carries its own records. The zone macros take only their
// fixed library types, so a zone with a metadata type of its own places its site through the metadata API,
// TT_DEBUG_SITE, and opens the zone on that site.

#include <cstdint>

#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "api/compute/compute_kernel_api.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "api/debug/kernel_profiler.hpp"
#include "hostdev/observability_event_id.h"

#include "tests/tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.h"

namespace m = observability_event_meta_silicon;

template <uint32_t Value>
void templated_zone() {
    TT_DEBUG_SITE(site, m::TemplatedMeta{m::kTemplatedName, Value});
    kernel_profiler::profileScope<site> zone;
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
    {
        TT_DEBUG_SITE(site, m::kNested);
        kernel_profiler::profileScope<site> zone;
    }
    {
        TT_DEBUG_SITE(site, m::kDeep);
        kernel_profiler::profileScope<site> zone;
    }
    {
        TT_DEBUG_SITE(site, m::kAllWidths);
        kernel_profiler::profileScope<site> zone;
    }
    {
        TT_DEBUG_SITE(site, m::kDefaulted);
        kernel_profiler::profileScope<site> zone;
    }
    {
        TT_DEBUG_SITE(site, m::kTwoStrings);
        kernel_profiler::profileScope<site> zone;
    }
    templated_zone<m::kTemplatedA>();
    templated_zone<m::kTemplatedB>();
    templated_zone<m::kTemplatedA>();
}

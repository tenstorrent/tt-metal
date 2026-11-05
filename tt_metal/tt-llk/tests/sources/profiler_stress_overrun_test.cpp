// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "profiler.h"

// Before this fix, is_buffer_full() reserved only 1 word per open zone, but each zone end is 2 words and the
// zone destructor wrote it unconditionally without a capacity check. Thus, when several zones were open
// near a full buffer, closing them pushed write_idx past the 1024-word buffer into the neighboring
// math buffer because the buffers are contiguous in memory.
//
//   501 fillers -> write_idx = 2*501 = 1002 (a zone writes nothing at open and reserves its 4 words).
//   With the enclosing kernel zone's 4 words reserved, 4 nested zones open before the guard blocks the next
//   one, and their closes plus the kernel zone's end at write_idx = 1022, inside the buffer.
//
// FILLER_COUNT / NEST_DEPTH are injected from the Python test (OVERRUN_FILL) via build.h.

#ifdef LLK_TRISC_UNPACK

static void open_nested_zones(std::uint32_t depth)
{
    if (depth == 0)
    {
        return;
    }
    ZONE_SCOPED("NEST");
    open_nested_zones(depth - 1);
}

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    for (std::uint32_t i = 0; i < FILLER_COUNT; i++)
    {
        TIMESTAMP("FILLER");
    }
    open_nested_zones(NEST_DEPTH);
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

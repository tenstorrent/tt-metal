// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "profiler.h"

struct RuntimeParams
{
};

// Before this fix, is_buffer_full() reserved only 1 word per open zone, but each zone end is 2 words and the
// zone destructor wrote it unconditionally without a capacity check. Thus, when several zones were open
// near a full buffer, closing them pushed write_idx past the 1024-word buffer into the neighboring
// math buffer because the buffers are contiguous in memory.
//
//   501 fillers -> write_idx = 2 (kernel start) + 2*501 = 1004.
//   The old guard then let ~6 nested zones open before blocking further opens; unwinding their
//   closes (plus the enclosing kernel zone) drove write_idx to ~1030 -> 6 words spilled into
//   the math buffer.
constexpr std::uint32_t FILLER_COUNT = 501;
constexpr std::uint32_t NEST_DEPTH   = 20;

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

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
    for (std::uint32_t i = 0; i < FILLER_COUNT; i++)
    {
        TIMESTAMP("FILLER");
    }
    open_nested_zones(NEST_DEPTH);
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
}

#endif

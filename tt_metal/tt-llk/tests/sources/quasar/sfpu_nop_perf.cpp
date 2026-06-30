// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

enum class PerfRunType
{
    MATH_ISOLATE,
};

#include "params.h"
#include "profiler.h"

namespace
{
constexpr std::uint32_t NOP_COUNT = 1000;

inline void mark_perf_run_type_used()
{
    (void)PERF_RUN_TYPE;
}

inline void run_empty_profiled_kernel()
{
    mark_perf_run_type_used();
    {
        ZONE_SCOPED("INIT")
        ckernel::tensix_sync();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        ckernel::tensix_sync();
    }
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    run_empty_profiled_kernel();
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    mark_perf_run_type_used();
    {
        ZONE_SCOPED("INIT")
        ckernel::tensix_sync();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t i = 0; i < NOP_COUNT; ++i)
        {
            asm volatile("nop");
        }
        ckernel::tensix_sync();
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    run_empty_profiled_kernel();
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    run_empty_profiled_kernel();
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU where (ternary): UNPACK2 (UNP_S) x3 -> SrcS -> SFPU -> PACK1 -> L1.
// All logic runs in LLK_TRISC_ISOLATE_SFPU through the WhereSrcs op type; UNPACK, MATH, PACK only open the
// perf zones so the zone rendezvous lines up across the four threads.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"

// PROFILER_SYNC() calls tensix_sync() unqualified.
using namespace ckernel;

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "llk_sfpu/ckernel_sfpu_srcs.h"
#include "llk_sfpu/ckernel_sfpu_where.h"
#include "llk_sfpu_srcs_api.h"
#include "params.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // condition
    const Operand& buffer_B         = params.buffer_B;   // true value
    const Operand& buffer_C         = params.buffer_C;   // false value
    const Operand& buffer_Res       = params.buffer_Res; // where(condition, true value, false value)
#endif
    const DataFormat unpack_S_dst = static_cast<DataFormat>(formats.unpack_S_dst);
    const DataFormat pack_S_src   = static_cast<DataFormat>(formats.pack_S_src);

    {
        START_PERF_MEASURE("INIT")
        // The SrcS layout is known only at runtime, so init and the tile loop each dispatch once.
        dispatch_sfpu_srcs_op<WhereSrcs>(
            unpack_S_dst,
            pack_S_src,
            [&](auto op)
            {
                using Op = decltype(op);
                Op::init(
                    L1_ADDRESS(buffer_A[0]),
                    L1_ADDRESS(buffer_B[0]),
                    L1_ADDRESS(buffer_C[0]),
                    static_cast<DataFormat>(formats.unpack_S_src),
                    L1_ADDRESS(buffer_Res[0]),
                    static_cast<DataFormat>(formats.pack_S_dst));
            });
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            dispatch_sfpu_srcs_op<WhereSrcs>(
                unpack_S_dst,
                pack_S_src,
                [&](auto op)
                {
                    using Op = decltype(op);
                    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                    {
                        Op::run(TILE_CNT);
                    }
                });
        }
        wait_sfpu_idle();
        wait_unpack_idle();
        wait_pack_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif

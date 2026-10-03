// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole compute semaphore ring protocol (api/compute/experimental/semaphore_compute_impl.h,
// COMPUTE_ATOMIC scope) on a datacopy pipeline of NUM_TILES_IN_BLOCK tiles per DEST section: per tile PACK
// wait_not_full(1), pack, up(1) and UNPACK wait_min(1), copy, down(1), RING_DEPTH credits seeded at init (0: no
// semaphore instructions). Semaphore 6 stands in for the API's index 3, which the harness's zone barrier uses.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint8_t RING_SEM     = semaphore::UNPACK_MATH_DONE;
constexpr std::uint8_t RING_SEM_MAX = 15;
constexpr bool RING                 = RING_DEPTH > 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void ring_wait_min_1()
{
    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(RING_SEM);
}

inline void ring_down_1()
{
    t6_semaphore_get<p_stall::UNPACK>(RING_SEM);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tiles       = params.NUM_TILES_IN_BLOCK;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<false>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            if constexpr (RING)
            {
                for (std::uint32_t i = 0; i < LOOP_FACTOR * tiles; ++i)
                {
                    ring_wait_min_1();
                    ring_down_1();
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, false>(4 * tiles * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < tiles; ++i)
                {
                    if constexpr (RING)
                    {
                        ring_wait_min_1();
                    }
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                        L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                    if constexpr (RING)
                    {
                        ring_down_1();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tiles       = params.NUM_TILES_IN_BLOCK;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
        _llk_math_hw_configure_<false>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, false, BroadcastType::NONE>(4, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true, true>(4 * tiles * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                for (std::uint32_t tile = 0; tile < tiles; ++tile)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, false, BroadcastType::NONE, false>(
                        tile, formats.math, formats.math);
                }
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

inline void ring_wait_not_full_1()
{
    t6_semaphore_wait_on_max<p_stall::STALL_PACK>(RING_SEM);
}

inline void ring_up_1()
{
    t6_semaphore_post<p_stall::PACK>(RING_SEM);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tiles       = params.NUM_TILES_IN_BLOCK;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<false, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_dest_init_<DstSync::SyncHalf, false>();
        if constexpr (RING)
        {
            t6_semaphore_init(RING_SEM, RING_DEPTH, RING_SEM_MAX);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            if constexpr (RING)
            {
                for (std::uint32_t i = 0; i < LOOP_FACTOR * tiles; ++i)
                {
                    ring_wait_not_full_1();
                    ring_up_1();
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < tiles; ++tile)
                {
                    if constexpr (RING)
                    {
                        ring_wait_not_full_1();
                    }
                    _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                    if constexpr (RING)
                    {
                        ring_up_1();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < tiles; ++tile)
                {
                    if constexpr (RING)
                    {
                        ring_wait_not_full_1();
                    }
                    _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                    if constexpr (RING)
                    {
                        ring_up_1();
                    }
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

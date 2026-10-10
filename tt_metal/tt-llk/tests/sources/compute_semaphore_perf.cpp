// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the compute semaphore ring (semaphore_compute_impl.h) on a datacopy pipeline; RING_DEPTH 0 runs no
// semaphore instructions. UNPACK_MATH_DONE stands in for the API's UNPACK_OPERAND_SYNC, which the harness's zone barrier
// uses. Max is the ring depth as in the API, and Value starts there: unpack runs at most RING_DEPTH tiles ahead of pack.

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

constexpr std::uint8_t RING_SEM = semaphore::UNPACK_MATH_DONE;
constexpr bool RING             = RING_DEPTH > 0;
static_assert(RING_DEPTH <= 15, "a semaphore's Value and Max are four bits");

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            if constexpr (RING)
            {
                for (std::uint32_t i = 0; i < LOOP_FACTOR * TILE_CNT; ++i)
                {
                    ring_wait_min_1();
                    ring_down_1();
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(TILE_NUM_FACES * TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    if constexpr (RING)
                    {
                        ring_wait_min_1();
                    }
                    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    LLK_ASSERT(
        (TILE_CNT <= get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "A DEST section holds more tiles than half of DEST");
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(TILE_NUM_FACES, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(TILE_NUM_FACES * TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        tile, formats.math, formats.math);
                }
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    LLK_ASSERT(
        (TILE_CNT <= get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "A DEST section holds more tiles than half of DEST");
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT /* tile_size */, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        if constexpr (RING)
        {
            t6_semaphore_init(RING_SEM, RING_DEPTH /*value*/, RING_DEPTH /*max*/);
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
                for (std::uint32_t i = 0; i < LOOP_FACTOR * TILE_CNT; ++i)
                {
                    ring_wait_not_full_1();
                    ring_up_1();
                }
            }
        }
        else
        {
            constexpr bool SYNC_WITH_MATH = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (SYNC_WITH_MATH)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (RING)
                    {
                        ring_wait_not_full_1();
                    }
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
                    if constexpr (RING)
                    {
                        ring_up_1();
                    }
                }
                if constexpr (SYNC_WITH_MATH)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

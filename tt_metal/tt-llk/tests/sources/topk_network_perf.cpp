// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// MATH_ISOLATE cost of one topk bitonic network API call (local sort / merge / rebuild) on a
// 2-tile slab already resident in DEST. One call per "tile", so TILE_LOOP reads as cycles per call.
// The network's timing does not depend on the data (every SFPSWAP issues regardless of its
// outcome), so DEST is not initialised.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE, "topk_network_perf measures MATH_ISOLATE only");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        _perf_unpack_loop_set_valid<true, false>(TILE_CNT * LOOP_FACTOR);
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_topk.h"

using namespace ckernel;
using namespace ckernel::sfpu;

constexpr bool TOPK_LARGEST   = (TOPK_IDIR == 0);
constexpr auto TOPK_TIE_ORDER = TOPK_STABLE_SORT ? (TOPK_LARGEST ? TopkTieOrder::Descending : TopkTieOrder::Ascending) : TopkTieOrder::Unset;

inline void topk_network_call()
{
    if constexpr (TOPK_NETWORK_OP == 0)
    {
        _bitonic_topk_phases_steps<true, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            TOPK_IDIR, TOPK_LOGK - 1, 0, 0, 0);
    }
    else if constexpr (TOPK_NETWORK_OP == 1)
    {
        _bitonic_topk_merge<true, is_fp32_dest_acc_en, TOPK_IDIR != 0, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(0, TOPK_K);
    }
    else
    {
        _bitonic_topk_rebuild<true, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            TOPK_IDIR != 0, 0, TOPK_K, TOPK_LOGK, TOPK_REBUILD_SKIP_SECOND);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
        if constexpr (TOPK_FUSED_STABLE)
        {
            _init_topk_fused_();
        }
        else if constexpr (TOPK_RANK_STAMPED)
        {
            _init_topk_rank_stamped_<16>();
        }
        else
        {
            _init_topk();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        _llk_math_eltwise_sfpu_start_(0);
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                topk_network_call();
                TTI_CLEARDVALID(1, 0);
            }
        }
        _llk_math_eltwise_sfpu_done_();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif

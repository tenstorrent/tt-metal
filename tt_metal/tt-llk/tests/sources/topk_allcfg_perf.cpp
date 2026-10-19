// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Scratch, not for merge: MATH_ISOLATE cost of one TopK LLK call per "tile", every call of the compute API
// (topk.h) at the arguments its callers use. Derived from topk_network_perf.cpp; compiles against main's and the
// stack's ckernel_sfpu_topk.h (TOPK_TILE0 is head only). The call is flattened as compute kernels inline it.

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

static_assert(PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE, "topk_allcfg_perf measures MATH_ISOLATE only");

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

// 0 sort, 1 merge, 2 rebuild, 3 fuse, 4 defuse, 5 stamp local, 6 stamp rank range, 7 canonicalize -0, 8 strip tags,
// 9 finalize uint16 indices, 10 uint16 prepare for pack, 11 uint16 move to pack half
inline __attribute__((flatten)) void topk_call()
{
    // m_iter is a loop variable in every caller, so keep it out of constant folding
    int m_iter = TOPK_MITER;
    asm volatile("" : "+r"(m_iter));
    if constexpr (TOPK_OP == 0)
    {
#if TOPK_TILE0
        _bitonic_topk_phases_steps_tile0_sorted_<true, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            TOPK_IDIR, TOPK_END_PHASE, 0, 0, 0);
#else
        _bitonic_topk_phases_steps<true, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            TOPK_IDIR, TOPK_END_PHASE, 0, 0, 0);
#endif
    }
    else if constexpr (TOPK_OP == 1)
    {
        _bitonic_topk_merge<true, is_fp32_dest_acc_en, TOPK_IDIR != 0, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            m_iter, TOPK_K);
    }
    else if constexpr (TOPK_OP == 2)
    {
        _bitonic_topk_rebuild<true, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER>(
            TOPK_IDIR != 0, m_iter, TOPK_K, TOPK_LOGK, TOPK_REBUILD_SKIP_SECOND);
    }
    else if constexpr (TOPK_OP == 3)
    {
        _topk_fuse_tile_<TOPK_LARGEST>();
    }
    else if constexpr (TOPK_OP == 4)
    {
        _topk_defuse_tile_<TOPK_LARGEST>(1);
    }
    else if constexpr (TOPK_OP == 5)
    {
        _topk_stamp_local_positions_<TOPK_LARGEST, 16>();
    }
    else if constexpr (TOPK_OP == 6)
    {
        _topk_stamp_tile_rank_range_<TOPK_LARGEST, 16>(1, 32);
    }
    else if constexpr (TOPK_OP == 7)
    {
        _topk_canonicalize_negzero_value_tiles_();
    }
    else if constexpr (TOPK_OP == 8)
    {
        _topk_strip_rank_tags_<16>(0);
    }
    else if constexpr (TOPK_OP == 9)
    {
        _topk_finalize_hi16_index_tile_(2);
    }
    else if constexpr (TOPK_OP == 10)
    {
        topk_uint16_prepare_value_tile_for_pack(0);
    }
    else
    {
        _topk_uint16_move_dest_tile_to_pack_half_(0);
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
                topk_call();
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

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// MATH_ISOLATE perf driver for the topk_xl index splits
// (sfpu/experimental/ckernel_sfpu_topk_xl.h). Blackhole-only.
//
// One "tile" of TILE_LOOP is one call of the selected split on Dst index 0, i.e.
// one K-element slot: the value region at Dst tile 0 and the index region one
// sequence further on. The splits are pure bit shuffles, so their cost does not
// depend on what Dest holds and no unpack or pack traffic is needed; unpack and
// pack return straight away and the marker covers the math thread only.
// PROFILER_SYNC (a tensix_sync) closes the marker, so the queued SFPU work is
// inside it.
//
// Selected by the TOPK_XL_SPLIT template parameter (TopKXLSplit):
//   0 RowMajor    _topk_xl_separate_indices_row_major_<K>, chunk base 0
//   1 Global      _topk_xl_separate_indices_row_major_global_<K>
//   2 GlobalBase  _topk_xl_separate_indices_row_major_global_base_<K>, seg_base = 32 * K
//   3 Separate    _topk_xl_separate_indices_<K, gid> (control: no row-major decode)

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

static_assert(PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE, "topk_xl_split_perf only implements MATH_ISOLATE");

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
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

#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

constexpr std::uint32_t SPLIT_GROUP_ID    = 0x2A9;
constexpr std::uint32_t SPLIT_GROUP_SHIFT = 16;
constexpr std::uint32_t SPLIT_SEG_BASE    = 32 * TOPK_XL_K;

constexpr bool SPLIT_ROW_MAJOR   = TOPK_XL_SPLIT == 0;
constexpr bool SPLIT_GLOBAL      = TOPK_XL_SPLIT == 1;
constexpr bool SPLIT_GLOBAL_BASE = TOPK_XL_SPLIT == 2;
constexpr bool SPLIT_SEPARATE    = TOPK_XL_SPLIT == 3;
static_assert(SPLIT_ROW_MAJOR || SPLIT_GLOBAL || SPLIT_GLOBAL_BASE || SPLIT_SEPARATE, "unknown TOPK_XL_SPLIT");

inline void split_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    if constexpr (SPLIT_SEPARATE)
    {
        ckernel::sfpu::_topk_xl_separate_indices_init_(SPLIT_GROUP_SHIFT);
    }
    else if constexpr (SPLIT_ROW_MAJOR)
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<0, 0>();
    }
    else
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
    }
}

inline void split_call()
{
    if constexpr (SPLIT_SEPARATE)
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_<TOPK_XL_K, SPLIT_GROUP_ID>, 0, VectorMode::RC_custom);
    }
    else if constexpr (SPLIT_GLOBAL_BASE)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<TOPK_XL_K>, 0, VectorMode::RC_custom, SPLIT_SEG_BASE);
    }
    else if constexpr (SPLIT_GLOBAL)
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<TOPK_XL_K>, 0, VectorMode::RC_custom);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_<TOPK_XL_K>, 0, VectorMode::RC_custom);
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
        _llk_math_pack_sync_init_<DstSync::SyncFull, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        split_init();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                split_call();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
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

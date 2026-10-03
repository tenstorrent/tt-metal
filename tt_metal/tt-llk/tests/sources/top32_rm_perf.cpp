// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MATH_ISOLATE perf driver for the DeepSeek top32_rm SFPU kernels
// (sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h), Blackhole only.
//
// TOP32_PERF_KERNEL picks one entry point; the math thread calls it LOOP_FACTOR times on Dest
// tile 0 (values in tiles 0/1, indices in tiles 2/3), through the same
// _llk_math_eltwise_unary_sfpu_params_(..., VectorMode::RC_custom, ...) call the compute
// kernels make, so the per-call Dest-base programming is part of what is measured. The sort is
// a fixed compare-exchange network, so its timing does not depend on the Dest contents, and
// nothing is unpacked or packed: the unpacker only hands out the valids the math loop clears.
//
//   0 phases_steps(descending)   1 phases_steps(ascending)
//   2 merge(across_tiles=false)  3 merge(across_tiles=true)
//   4 rebuild(desc, skip_second) 5 rebuild(asc, skip_second)  6 rebuild(desc, full)
//   7 pre_sorted_prep<desc>      8 pre_sorted_prep<asc>
//   9 pre_sorted_combine         10 pre_sorted_final

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE, "top32_rm_perf only implements MATH_ISOLATE");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
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
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"

using namespace ckernel;
using namespace ckernel::sfpu;

static constexpr std::uint32_t DST_TILE = 0;
static_assert(TOP32_PERF_KERNEL <= 10, "unknown TOP32_PERF_KERNEL");

inline void run_top32_kernel()
{
    constexpr bool APPROX = false;
    if constexpr (TOP32_PERF_KERNEL == 0)
    {
        _llk_math_eltwise_unary_sfpu_params_(_bitonic_top32_phases_steps_<APPROX, is_fp32_dest_acc_en>, DST_TILE, VectorMode::RC_custom, 0);
    }
    else if constexpr (TOP32_PERF_KERNEL == 1)
    {
        _llk_math_eltwise_unary_sfpu_params_(_bitonic_top32_phases_steps_<APPROX, is_fp32_dest_acc_en>, DST_TILE, VectorMode::RC_custom, 1);
    }
    else if constexpr (TOP32_PERF_KERNEL == 2 || TOP32_PERF_KERNEL == 3)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_top32_merge_<APPROX, is_fp32_dest_acc_en, false>, DST_TILE, VectorMode::RC_custom, TOP32_PERF_KERNEL == 3);
    }
    else if constexpr (TOP32_PERF_KERNEL >= 4 && TOP32_PERF_KERNEL <= 6)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_top32_rebuild_<APPROX, is_fp32_dest_acc_en>, DST_TILE, VectorMode::RC_custom, TOP32_PERF_KERNEL == 5, TOP32_PERF_KERNEL != 6);
    }
    else if constexpr (TOP32_PERF_KERNEL == 7 || TOP32_PERF_KERNEL == 8)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_top32_of_1024_rm_pre_sorted_prep_ < APPROX, is_fp32_dest_acc_en, TOP32_PERF_KERNEL == 8 >, DST_TILE, VectorMode::RC_custom, DST_TILE);
    }
    else if constexpr (TOP32_PERF_KERNEL == 9)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_top32_of_1024_rm_pre_sorted_combine_<APPROX, is_fp32_dest_acc_en>, DST_TILE, VectorMode::RC_custom, DST_TILE);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_top32_of_1024_rm_pre_sorted_final_<APPROX, is_fp32_dest_acc_en>, DST_TILE, VectorMode::RC_custom, DST_TILE);
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
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        _top32_rm_init_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                run_top32_kernel();
                TTI_CLEARDVALID(1, 0);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

using namespace ckernel;

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

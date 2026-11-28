// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// topk_xl split sorts (sfpu/experimental/ckernel_sfpu_topk_xl_halves.h, _quarters.h): one 2048-element row sorted
// as halves or quarters and combined into the stock topk_xl 2048 order. Blackhole only.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(TOPK_XL_PARTS == 2 || TOPK_XL_PARTS == 4, "the row splits into two halves or four quarters");
static_assert(TOPK_XL_PARTS_CORE_ID < 32, "the core id fills index bits [15:11]");
static_assert(is_fp32_dest_acc_en, "the value|index words are 32-bit Dest words");
static_assert(
    PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE,
    "the SFPU work runs on MATH: measure it end to end or with MATH isolated");

constexpr std::uint32_t ROW_ELEMENTS  = 2048;
constexpr std::uint32_t CORE_ID_SHIFT = 11;
constexpr bool QUARTERS               = TOPK_XL_PARTS == 4;
constexpr std::uint32_t PART_ELEMENTS = ROW_ELEMENTS / TOPK_XL_PARTS;
constexpr bool COMBINE                = TOPK_XL_PARTS_COMBINE;
constexpr std::uint32_t OUT_TILE      = 0;
// _topk_xl_combine_halves_ reads its two halves at OUT_TILE + 2 and + 3; quarter pair h combines into half tile h.
constexpr std::uint32_t HALF_TILE         = OUT_TILE + 2;
constexpr std::uint32_t FIRST_PART_TILE   = QUARTERS ? HALF_TILE + 2 : HALF_TILE;
constexpr std::uint32_t RESULT_TILES      = COMBINE ? 2 : TOPK_XL_PARTS;
constexpr std::uint32_t FIRST_RESULT_TILE = COMBINE ? OUT_TILE : FIRST_PART_TILE;
// Quarters use Dest tiles up to FIRST_PART_TILE + 3 = 7, which fit in Dest only with full sync.
static_assert(!QUARTERS || dest_sync == ckernel::DstSync::SyncFull, "the quarter tiles need the full Dest");
// Every local sort, half prep and combine ends by releasing SrcA and SrcB (CLR_AB) after face moves that wait for
// both to be valid; UNPACK supplies one dummy valid per release.
constexpr std::uint32_t SRC_RELEASES = TOPK_XL_PARTS + (COMBINE ? 3 : 0);

// Direction of part p's local sort. Halves: opposite. Quarters: opposite within each pair that one combine reads.
constexpr bool part_ascending(std::uint32_t p)
{
    return QUARTERS ? (((p & 1) == 0) == TOPK_XL_QUARTER0_ASCENDING) : ((p == 0) == TOPK_XL_HALF0_ASCENDING);
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h"
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif
    const std::uint32_t src_format = formats.unpack_A_src;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        ckernel::_llk_unpack_topk_xl_copy_init_(src_format, dst_format);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                _perf_unpack_stream_set_valid<true /*set_a*/, true /*set_b*/, PERF_RUN_TYPE>(TOPK_XL_PARTS + SRC_RELEASES);
            }
            else
            {
                TT_SETADCXX(p_setadc::UNP_A, PART_ELEMENTS - 1, 0x0 /*x_start*/);
                for (std::uint32_t p = 0; p < TOPK_XL_PARTS; p++)
                {
                    ckernel::_llk_unpack_topk_xl_copy_(L1_ADDRESS(buffer_A[p]), src_format, dst_format, PART_ELEMENTS);
                }
                TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0 /*x_start*/);
                for (std::uint32_t i = 0; i < SRC_RELEASES; i++)
                {
                    _llk_unpack_set_srcb_dummy_valid_();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl_halves.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl_quarters.h"
#include "topk_xl_scrub_dest.h"

using namespace ckernel;

template <std::uint32_t p>
inline void stamp_part()
{
    constexpr std::uint32_t tile = FIRST_PART_TILE + p;
    if constexpr (QUARTERS)
    {
        constexpr std::uint32_t base = (TOPK_XL_PARTS_CORE_ID << CORE_ID_SHIFT) | (p * PART_ELEMENTS);
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_add_lsb_indices_base_rt_<PART_ELEMENTS>, tile, VectorMode::RC_custom, base);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_add_lsb_indices_base_<TOPK_XL_PARTS_CORE_ID, p * PART_ELEMENTS>, tile, VectorMode::RC_custom);
    }
}

__attribute__((noinline)) void copy_stamp_sort(std::uint32_t math_format)
{
    _llk_math_topk_xl_copy_init_(math_format);
    for (std::uint32_t p = 0; p < TOPK_XL_PARTS; p++)
    {
        _llk_math_topk_xl_copy_(FIRST_PART_TILE + p, math_format, PART_ELEMENTS);
    }

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    sfpu::_topk_xl_add_lsb_indices_init_();
    stamp_part<0>();
    stamp_part<1>();
    if constexpr (QUARTERS)
    {
        stamp_part<2>();
        stamp_part<3>();
    }

    // The stamp init leaves ADDR_MOD_6 at +4 and ADDR_MOD_4 at +16. The local sorts and combines store through
    // ADDR_MOD_6 expecting the +32 that _topk_xl_init_ sets and do not reprogram it, so it must run again after the
    // stamp and before any of them.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    sfpu::_topk_xl_init_<PART_ELEMENTS, true /*fused*/>();
    for (std::uint32_t p = 0; p < TOPK_XL_PARTS; p++)
    {
        const std::uint32_t tile = FIRST_PART_TILE + p;
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_local_sort_ool_<PART_ELEMENTS>, tile, VectorMode::RC_custom, tile, part_ascending(p));
    }
}

__attribute__((noinline)) void combine()
{
    if constexpr (QUARTERS)
    {
        // Quarter pair h (parts 2h and 2h + 1) -> half tile h.
        constexpr std::uint32_t pair0 = FIRST_PART_TILE;
        constexpr std::uint32_t pair1 = FIRST_PART_TILE + 2;
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::_topk_xl_combine_quarters_<pair0, HALF_TILE>, pair0 /*dst_index*/, VectorMode::RC_custom, TOPK_XL_HALF0_ASCENDING);
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::_topk_xl_combine_quarters_<pair1, HALF_TILE + 1>, pair1 /*dst_index*/, VectorMode::RC_custom, !TOPK_XL_HALF0_ASCENDING);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_half_prep_, HALF_TILE /*dst_index*/, VectorMode::RC_custom, HALF_TILE);
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_half_prep_, HALF_TILE + 1 /*dst_index*/, VectorMode::RC_custom, HALF_TILE + 1);
    }
    _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_combine_halves_, OUT_TILE /*dst_index*/, VectorMode::RC_custom, OUT_TILE, TOPK_XL_PARTS_ASCENDING);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const std::uint32_t math_format = formats.math;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        topk_xl_scrub_dest();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
            }
            copy_stamp_sort(math_format);
            if constexpr (COMBINE)
            {
                combine();
            }
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }

    topk_xl_scrub_dest();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t t = 0; t < RESULT_TILES; t++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(FIRST_RESULT_TILE + t, L1_ADDRESS(buffer_Res[t]));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

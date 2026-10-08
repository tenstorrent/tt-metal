// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// topk_xl_perf.cpp's fused K = 512 and K = 2048 rows with the chunk split across threads: the SFPU work on PACK, the copy and
// the face transposes on MATH, two chunks in flight (helpers/include/topk_xl_split.h). Same unpack stream, same
// Dst result. L1_TO_L1 only, Blackhole-only.

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

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "topk_xl_split_perf runs L1_TO_L1 only");
static_assert((TOPK_XL_K == 512 || TOPK_XL_K == 2048) && TOPK_XL_FUSED_E2E, "the split pipeline covers the fused K = 512 and 2048 chunks");
static_assert(TOPK_XL_NUM_CHUNKS >= 2 && TOPK_XL_NUM_CHUNKS <= 32, "the perf rows merge 2 to 32 chunks");

constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
constexpr std::uint32_t TILES_PER_SEQ     = TOPK_XL_K / ELEMENTS_PER_TILE + (TOPK_XL_K < ELEMENTS_PER_TILE ? 1 : 0);
constexpr std::uint32_t TILE_ELEMENTS     = TOPK_XL_K < ELEMENTS_PER_TILE ? TOPK_XL_K : ELEMENTS_PER_TILE;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h"
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

__attribute__((noinline)) void unpack_copy_tile(RUNTIME_PARAMETERS params, std::uint32_t c, std::uint32_t src_format, std::uint32_t dst_format)
{
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        TT_SETADCXX(p_setadc::UNP_A, TILE_ELEMENTS - 1, 0x0);
        ckernel::_llk_unpack_topk_xl_copy_(L1_ADDRESS(params.buffer_A[c * TILES_PER_SEQ + t]), src_format, dst_format, TILE_ELEMENTS);
    }
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t src_format  = formats.unpack_A_src;
    const std::uint32_t dst_format  = formats.unpack_A_dst;
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
            for (std::uint32_t c = 0; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                unpack_copy_tile(params, c, src_format, dst_format);
                _llk_unpack_set_srcb_dummy_valid_(); // local sort
                if (c > 0)
                {
                    _llk_unpack_set_srcb_dummy_valid_(); // rebuild
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
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#include "topk_xl_split.h"

using namespace ckernel;

// ZEROACC only sets Dest's zero flags; write real zeros once PACK has released Dest, so no run sees another's leftovers.
static __attribute__((noinline)) void scrub_dest()
{
    ckernel::tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    }
    reset_dest_offset_id();
    math::set_dest_section_base<StartZero>();
    constexpr std::uint32_t dest_tiles = get_dest_max_tiles<DstSync::SyncFull, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
    for (std::uint32_t tile = 0; tile < dest_tiles; tile++)
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_int_<false, InstrModLoadStore::INT32, 8>, tile, VectorMode::RC, 0u);
    }
    ckernel::tensix_sync();
}

static __attribute__((noinline)) void copy_chunk(std::uint32_t tile, std::uint32_t dst_format)
{
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        ckernel::_llk_math_topk_xl_copy_(tile + t, dst_format, TILE_ELEMENTS);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t math_format = formats.math;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::_llk_math_topk_xl_copy_init_(math_format);
        topk_xl_split::init_tokens();
        PROFILER_SYNC();
    }
    scrub_dest();
    topk_xl_split::start_pack();
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            topk_xl_split::math_row<TOPK_XL_K>(TOPK_XL_NUM_CHUNKS, [math_format](std::uint32_t, std::uint32_t tile) { copy_chunk(tile, math_format); });
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
    scrub_dest();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#include "topk_xl_split.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    topk_xl_split::wait_start<TOPK_XL_K>();
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            topk_xl_split::pack_row<TOPK_XL_K>(TOPK_XL_NUM_CHUNKS);
            topk_xl_split::split_indices<TOPK_XL_K>(0);
            _llk_packer_wait_for_math_done_();
            _llk_pack_mop_config_<PackMode::Default, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
            // The value region of sequence 0, then its index region after it.
            for (std::uint32_t t = 0; t < 2 * TILES_PER_SEQ; t++)
            {
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(t, L1_ADDRESS(params.buffer_Res[t]));
            }
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf twin of topk_xl_test.cpp: the two topk_large_indices chunk pipelines of one row (copy, index stamp, local sort,
// merge and rebuild per chunk, then the index split), repeated LOOP_FACTOR times inside TILE_LOOP and reported per input
// tile. TOPK_XL_FUSED_E2E selects the fused end-to-end path (runtime chunk-id stamp, fused merge and rebuild, one global
// split); otherwise the unfused row-major op path. L1_TO_L1 only: the topk_xl copy has no unpack mock. Blackhole-only.

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

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "topk_xl_perf runs L1_TO_L1 only");

constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
constexpr std::uint32_t TILES_PER_SEQ     = (TOPK_XL_K + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
constexpr std::uint32_t SLOT0             = 0;
// Second merge operand: one tile per sequence tile when fused, the value and index regions when unfused.
constexpr std::uint32_t SLOT1 = TOPK_XL_FUSED_E2E ? TILES_PER_SEQ : (2 * TILES_PER_SEQ);

static_assert(TOPK_XL_NUM_CHUNKS >= 2, "the perf rows merge at least two chunks");

inline constexpr std::uint32_t tile_active_elements(std::uint32_t t)
{
    return (t == 0) ? ((TOPK_XL_K < ELEMENTS_PER_TILE) ? TOPK_XL_K : ELEMENTS_PER_TILE) : ((TOPK_XL_K > ELEMENTS_PER_TILE) ? (TOPK_XL_K - ELEMENTS_PER_TILE) : 0);
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h"
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

__attribute__((noinline)) void unpack_copy_tile(RUNTIME_PARAMETERS params, std::uint32_t c, std::uint32_t src_format, std::uint32_t dst_format)
{
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        const std::uint32_t elements = tile_active_elements(t);
        TT_SETADCXX(p_setadc::UNP_A, (elements == 0) ? (ELEMENTS_PER_TILE - 1) : (elements - 1), 0x0);
        ckernel::_llk_unpack_topk_xl_copy_(L1_ADDRESS(params.buffer_A[c * TILES_PER_SEQ + t]), src_format, dst_format, elements);
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

// TRISC1 code region overflows under the default -O3.
#pragma GCC optimize("O2")

#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

using namespace ckernel;

template <bool fused>
inline void topk_xl_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_init_<TOPK_XL_K, fused>();
}

template <bool fused>
__attribute__((noinline)) void merge_and_rebuild()
{
    topk_xl_init<fused>();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_merge_<TOPK_XL_K, fused>, SLOT0, VectorMode::RC_custom, SLOT0);
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_rebuild_<TOPK_XL_K, fused>, SLOT0, VectorMode::RC_custom, SLOT0, false /* ascending */);
}

static __attribute__((noinline)) void copy_chunk(std::uint32_t slot, std::uint32_t dst_format)
{
    ckernel::_llk_math_topk_xl_copy_init_(dst_format);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        ckernel::_llk_math_topk_xl_copy_(slot + t, dst_format, tile_active_elements(t));
    }
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_add_lsb_indices_init_();
}

// Fused end to end: the chunk id stamped at run time into index bits [15:11].
static __attribute__((noinline)) void copy_sort_rt(std::uint32_t slot, bool ascending, std::uint32_t chunk_id, std::uint32_t dst_format)
{
    copy_chunk(slot, dst_format);
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_add_lsb_indices_rt_<TOPK_XL_K>, slot, VectorMode::RC_custom, chunk_id);
    topk_xl_init<true>();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_local_sort_<TOPK_XL_K>, slot, VectorMode::RC_custom, slot, ascending);
}

// Row-major op path: sort, then split into unfused values and row-major u32 indices for the merge tree.
static __attribute__((noinline)) void process_chunk(std::uint32_t slot, bool ascending, std::uint32_t dst_format)
{
    copy_chunk(slot, dst_format);
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_add_lsb_indices_<TOPK_XL_K, 0, false>, slot, VectorMode::RC_custom);
    topk_xl_init<true>();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_local_sort_<TOPK_XL_K>, slot, VectorMode::RC_custom, slot, ascending);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_reinit_();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_<TOPK_XL_K>, slot, VectorMode::RC_custom);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_advance_chunk_base_<TOPK_XL_K>();
}

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
        PROFILER_SYNC();
    }
    scrub_dest();
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            if constexpr (TOPK_XL_FUSED_E2E)
            {
                copy_sort_rt(SLOT0, false /* ascending */, 0, math_format);
                for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
                {
                    copy_sort_rt(SLOT1, true /* ascending */, c, math_format);
                    merge_and_rebuild<true>();
                }
                ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
                _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<TOPK_XL_K>, SLOT0, VectorMode::RC_custom);
            }
            else
            {
                _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
                ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<0, 0>();
                process_chunk(SLOT0, false /* ascending */, math_format);
                for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
                {
                    process_chunk(SLOT1, true /* ascending */, math_format);
                    merge_and_rebuild<false>();
                }
            }
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
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_packer_wait_for_math_done_();
            // The value region of slot 0, then its index region.
            for (std::uint32_t t = 0; t < 2 * TILES_PER_SEQ; t++)
            {
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SLOT0 + t, L1_ADDRESS(params.buffer_Res[t]));
            }
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

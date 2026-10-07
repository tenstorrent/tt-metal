// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Functional twin of topk_xl_split_perf.cpp: topk_xl_test.cpp's fused end-to-end rows at K = 512, with the chunk
// split across threads (helpers/include/topk_xl_split.h). Same TOPK_XL parameters, stimuli and packed result as
// topk_xl_test.cpp, so the two outputs compare bit for bit. Blackhole-only.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(TOPK_XL_K == 512 && TOPK_XL_FUSED_E2E, "the split pipeline covers the fused K = 512 chunk only");
static_assert(TOPK_XL_NUM_CHUNKS >= 1 && TOPK_XL_NUM_CHUNKS <= 32, "the chunk id stamp holds 32 chunks");

inline constexpr std::uint32_t chunk_active_elements(std::uint32_t c)
{
    return (c == TOPK_XL_NUM_CHUNKS - 1) ? TOPK_XL_TAIL_ELEMENTS : TOPK_XL_K;
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h"
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

__attribute__((noinline)) void unpack_copy_tile(RUNTIME_PARAMETERS params, std::uint32_t r, std::uint32_t c, std::uint32_t src_format, std::uint32_t dst_format)
{
    const std::uint32_t active = chunk_active_elements(c);
    TT_SETADCXX(p_setadc::UNP_A, active - 1, 0x0);
    ckernel::_llk_unpack_topk_xl_copy_(L1_ADDRESS(params.buffer_A[r * TOPK_XL_NUM_CHUNKS + c]), src_format, dst_format, active);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t src_format = formats.unpack_A_src;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    ckernel::_llk_unpack_topk_xl_copy_init_(src_format, dst_format);
    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        for (std::uint32_t c = 0; c < TOPK_XL_NUM_CHUNKS; c++)
        {
            unpack_copy_tile(params, r, c, src_format, dst_format);
            _llk_unpack_set_srcb_dummy_valid_(); // local sort
            if (c > 0)
            {
                _llk_unpack_set_srcb_dummy_valid_(); // rebuild
            }
        }
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

static __attribute__((noinline)) void copy_chunk(std::uint32_t chunk, std::uint32_t tile, std::uint32_t dst_format)
{
    ckernel::_llk_math_topk_xl_copy_(tile, dst_format, chunk_active_elements(chunk));
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t math_format = formats.math;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::_llk_math_topk_xl_copy_init_(math_format);
    topk_xl_split::init_tokens();
    scrub_dest();
    topk_xl_split::start_pack();

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        topk_xl_split::math_row(
            TOPK_XL_NUM_CHUNKS, [math_format](std::uint32_t chunk, std::uint32_t tile) { copy_chunk(chunk, tile, math_format); });
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    topk_xl_split::wait_start();

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        topk_xl_split::pack_row(TOPK_XL_NUM_CHUNKS);
        topk_xl_split::split_indices<TOPK_XL_K>(TOPK_XL_SEG_BASE);
        _llk_packer_wait_for_math_done_();
        _llk_pack_mop_config_<PackMode::Default, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
        for (std::uint32_t t = 0; t < 2; t++)
        {
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(t, L1_ADDRESS(params.buffer_Res[2 * r + t]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_PACK

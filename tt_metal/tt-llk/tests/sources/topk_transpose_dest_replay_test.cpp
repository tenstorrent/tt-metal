// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for the topk replay-buffer contract: topk records over the replay slots
// transpose_dest recorded at init, so transpose_dest must re-run its init after topk.
//
// Tile 0: transpose_dest init, transpose. Tile 1: topk local sort on the next DEST half, then
// transpose_dest init again, transpose. Both tiles must come out transposed. Uses the 32-bit
// unpack-to-dest path, where nothing between the two transposes but topk touches transpose_dest state.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr auto DST_SYNC           = ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t NUM_TILES = 2;
static constexpr std::uint32_t DST_INDEX = 0;
static constexpr bool IS_32BIT_TRANSPOSE = true;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        params.UNPACK_TRANSPOSE_FACES, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < NUM_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
        // SrcB valid for the transpose's MOVD2B/MOVB2D sequence.
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// Set before the TopK LLK API header, as topk_test.cpp does.
#define DST_SYNC_MODE  DST_SYNC
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/ckernel_sfpu_topk.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

using namespace ckernel;

static_assert(unpack_to_dest && is_fp32_dest_acc_en, "the test needs the 32-bit unpack-to-dest transpose path");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto transpose_tile = [&]
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            DST_INDEX, formats.math, formats.math);
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, MATH_TRANSPOSE_FACES, IS_32BIT_TRANSPOSE>(DST_INDEX);
    };

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
    _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, IS_32BIT_TRANSPOSE>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    transpose_tile();
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    // Sorts whatever this DEST half holds; only its replay-buffer recording matters here.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_bitonic_topk_phases_steps<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en>,
        DST_INDEX,
        VectorMode::RC_custom,
        0 /* idir */,
        4 /* i_end_phase: log2(32) - 1 */,
        0 /* i_start_phase */,
        0 /* i_end_step */,
        0 /* i_start_step */);

    // Required: topk overwrote the replay slots the first init recorded.
    _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, IS_32BIT_TRANSPOSE>();
    transpose_tile();
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t tile = 0; tile < NUM_TILES; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

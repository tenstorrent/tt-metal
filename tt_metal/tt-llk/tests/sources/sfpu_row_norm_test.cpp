// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// L1 harness driver for hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_row_norm.h.
// Covers all entry points across NUM_ROWS in {1, 2, 4, 8} and TILE_H in {NUM_ROWS, 16, 32}:
//   OP_MODE == 0: _calculate_row_rmsnorm_
//   OP_MODE == 1: _calculate_row_reduce_scale_add_rsqrt_
//   OP_MODE == 2: _calculate_row_layernorm_
//   OP_MODE == 3: wide-row 3-stage layernorm (reduce_mean + sum_centered_squares + finalize_inv_std + center_scale)

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

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
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < ROW_NORM_NUM_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

#define DST_ACCUM_MODE is_fp32_dest_acc_en
constexpr bool APPROX = false;
#include "experimental/llk_sfpu/ckernel_sfpu_row_norm.h"
#undef DST_ACCUM_MODE

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_row_norm_init_<false /* APPROX */>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();

    for (std::uint32_t tile = 0; tile < ROW_NORM_NUM_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    if constexpr (ROW_NORM_OP_MODE == 0)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_rmsnorm_<
                false /* APPROX */,
                false /* FAST_APPROX */,
                is_fp32_dest_acc_en,
                ROW_NORM_NUM_ROWS,
                ROW_NORM_TILE_H,
                ROW_NORM_NUM_TILES,
                ROW_NORM_SCALE_SUM>,
            0 /* dst_index */,
            VectorMode::RC_custom,
            ROW_NORM_INV_WIDTH_BITS,
            ROW_NORM_EPSILON_BITS);
    }
    else if constexpr (ROW_NORM_OP_MODE == 1)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_reduce_scale_add_rsqrt_<
                false /* APPROX */,
                false /* FAST_APPROX */,
                is_fp32_dest_acc_en,
                ROW_NORM_NUM_ROWS,
                ROW_NORM_TILE_H,
                ROW_NORM_NUM_TILES,
                ROW_NORM_SCALE_SUM,
                false /* FACE0_ONLY */>,
            0 /* dst_index */,
            VectorMode::RC_custom,
            ROW_NORM_INV_WIDTH_BITS,
            ROW_NORM_EPSILON_BITS);
    }
    else if constexpr (ROW_NORM_OP_MODE == 2)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_layernorm_<
                false /* APPROX */,
                false /* FAST_APPROX */,
                is_fp32_dest_acc_en,
                ROW_NORM_NUM_ROWS,
                ROW_NORM_TILE_H,
                ROW_NORM_NUM_TILES>,
            0 /* dst_index */,
            VectorMode::RC_custom,
            ROW_NORM_INV_WIDTH_BITS,
            ROW_NORM_EPSILON_BITS);
    }
    else
    {
        constexpr std::uint32_t kStatsTile = ROW_NORM_NUM_TILES;
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_reduce_mean_<is_fp32_dest_acc_en, ROW_NORM_NUM_ROWS, ROW_NORM_TILE_H, ROW_NORM_NUM_TILES, kStatsTile>,
            0 /* dst_index */,
            VectorMode::RC_custom,
            ROW_NORM_INV_WIDTH_BITS);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_sum_centered_squares_<
                is_fp32_dest_acc_en,
                ROW_NORM_NUM_ROWS,
                ROW_NORM_TILE_H,
                ROW_NORM_NUM_TILES,
                kStatsTile,
                false /* ACCUMULATE */>,
            0 /* dst_index */,
            VectorMode::RC_custom);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_finalize_inv_std_<false /* APPROX */, false /* FAST_APPROX */, is_fp32_dest_acc_en, ROW_NORM_NUM_ROWS, kStatsTile>,
            0 /* dst_index */,
            VectorMode::RC_custom,
            ROW_NORM_INV_WIDTH_BITS,
            ROW_NORM_EPSILON_BITS);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_row_center_scale_<is_fp32_dest_acc_en, ROW_NORM_NUM_ROWS, ROW_NORM_TILE_H, ROW_NORM_NUM_TILES, kStatsTile>,
            0 /* dst_index */,
            VectorMode::RC_custom);
    }

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

    _llk_packer_wait_for_math_done_();
    constexpr std::uint32_t kPackTiles = (ROW_NORM_OP_MODE == 1) ? 1 : ROW_NORM_NUM_TILES;
    for (std::uint32_t tile = 0; tile < kPackTiles; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

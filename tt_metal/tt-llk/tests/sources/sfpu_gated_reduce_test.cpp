// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t block_tiles = 4;
constexpr auto DST_SYNC             = dest_sync;

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
        0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_gated_reduce.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    constexpr int iterations   = GATED_REDUCE_ROWS <= 4 ? 2 : GATED_REDUCE_ROWS <= 8 ? 4 : 8;
    constexpr auto vector_mode = GATED_REDUCE_ROWS <= 16 ? VectorMode::R : VectorMode::RC;
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += block_tiles)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t tile = 0; tile < block_tiles; ++tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }
        if (base != 0)
        {
            // Approximate exp overwrites LREG12 (vConstFloatPrgm0). Reinitializing
            // sigmoid must restore the 2.0 constant used by its reciprocal.
            sfpu::exp_init<true, 0x3f800000 /* scale = 1.0f */, true, is_fp32_dest_acc_en>();
        }
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::silu>();
        sfpu::sigmoid_init<false>();
        const auto block     = base / block_tiles;
        const auto scale     = block == 2 ? params.GATED_OUT_SCALE_BITS : params.GATED_SCALE_BITS;
        const auto out_scale = block == 2 ? params.GATED_SCALE_BITS : params.GATED_OUT_SCALE_BITS;
        // First block has two experts; next has guards on both sides; last is a partial batch.
        for (std::uint32_t gate = block == 0 ? 0 : block; gate + 1 < block_tiles; gate += 2)
        {
            _llk_math_eltwise_unary_sfpu_params_(
                sfpu::calculate_gated_reduce < GATED_REDUCE_GATE,
                GATED_REDUCE_UP,
                (GATED_REDUCE_SCALE_FLAGS & 1) != 0,
                (GATED_REDUCE_SCALE_FLAGS & 2) != 0,
                (GATED_REDUCE_SCALE_FLAGS & 4) != 0,
                is_fp32_dest_acc_en,
                iterations >
                , gate, vector_mode, scale, out_scale, params.GATED_LIMIT_BITS, params.GATED_ALPHA_BITS);
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += block_tiles)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < block_tiles; ++tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[base + tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}
#endif

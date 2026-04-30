// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, params.num_faces, formats.unpack_A_src, formats.unpack_A_dst);

    const int num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (int tile = 0; tile < num_total_tiles; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(params.num_faces, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(params.num_faces, formats.math);
#endif
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                ((params.DST_INDEX + tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
#ifdef ARCH_BLACKHOLE
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile, formats.math, formats.math, params.num_faces);
#else
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile, formats.math, formats.math);
#endif
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, tilize_en>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, params.num_faces, false, false, params.RELU_CONFIG);
    _llk_pack_init_<false, false, tilize_en>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, params.num_faces, false, false, params.RELU_CONFIG);
    _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#endif
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            std::uint32_t res_tile_idx = block * num_tiles_in_block + tile;
            LLK_ASSERT(
                ((params.DST_INDEX + tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(params.DST_INDEX + tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}
#endif

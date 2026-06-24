// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "profiler.h"

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
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (std::uint32_t tile = 0; tile < num_total_tiles; ++tile)
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
#include "llk_sfpu/ckernel_sfpu_reduce.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        4 /* num_faces */, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();
    ckernel::sfpu::init_reduce<POOL_TYPE, static_cast<DataFormat>(formats.math), is_fp32_dest_acc_en>();

    if (REDUCE_DIM == ReduceDim::REDUCE_COL)
    {
        // Column reduction can be done block by block
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
            {
                LLK_ASSERT(
                    (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile, formats.math, formats.math);
            }

            for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
            {
                LLK_ASSERT(
                    (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_math_eltwise_sfpu_start_(tile);
                ckernel::sfpu::calculate_reduce<
                    POOL_TYPE,
                    REDUCE_DIM,
                    static_cast<DataFormat>(formats.math),
                    is_fp32_dest_acc_en,
                    static_cast<DataFormat>(formats.pack_dst)>();
            }

            _llk_math_eltwise_sfpu_done_();
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
    }
    else if (REDUCE_DIM == ReduceDim::REDUCE_ROW)
    {
        // Row reduction requires all tiles in destination at once
        LLK_ASSERT(num_blocks == 1, "Row reduction requires all tiles in one block");
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t i = 0; i < num_tiles_in_block * num_blocks; ++i)
        {
            LLK_ASSERT(
                (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                i, formats.math, formats.math);
        }

        _llk_math_eltwise_sfpu_start_(0);
        ckernel::sfpu::
            calculate_reduce<POOL_TYPE, REDUCE_DIM, static_cast<DataFormat>(formats.math), is_fp32_dest_acc_en, static_cast<DataFormat>(formats.pack_dst)>(
                BLOCK_CT_DIM, BLOCK_RT_DIM);

#ifdef ADD_TOP_ROW
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::add_top_row>();
        _llk_math_eltwise_sfpu_start_(0);
        ckernel::sfpu::_init_add_top_row_();

        for (int i = 1; i < num_tiles_in_block * num_blocks; ++i)
        {
            LLK_ASSERT(
                (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            // Add the top rows of all the tiles we reduced in dst register
            ckernel::sfpu::_calculate_add_top_row_<static_cast<DataFormat>(formats.math)>(0, i, 0); // accumulate the result in tile at index 0
        }
#endif

        _llk_math_eltwise_sfpu_done_();
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            std::uint32_t res_tile_idx = (block * num_tiles_in_block) + tile;
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

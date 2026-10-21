// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// UNPACK_BLOCK (driver template): the block calls of tilize_block instead of one call per tile
#ifndef UNPACK_BLOCK
#define UNPACK_BLOCK 0
#endif

#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE)
// As tilize_block: four-face tiles unpacked to a 32-bit DEST take one DEST section and one handshake per run of a row's tiles
constexpr bool DEST_RUNS               = unpack_to_dest && is_fp32_dest_acc_en;
constexpr std::uint32_t DEST_RUN_TILES = 4; // SyncHalf, 32-bit DEST
#else
constexpr bool DEST_RUNS               = false;
constexpr std::uint32_t DEST_RUN_TILES = 0;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces = params.num_faces;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
    _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, params.BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */, num_faces, UNPACK_BLOCK);

    std::uint32_t read_offset = 0;

    const std::uint32_t block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(params.BLOCK_CT_DIM);

    for (std::uint32_t i = 0; i < params.BLOCK_RT_DIM; i++)
    {
#if UNPACK_BLOCK
#if defined(ARCH_BLACKHOLE)
        if (DEST_RUNS && num_faces == 4)
        {
            _llk_unpack_tilize_dest_block_(
                L1_ADDRESS(params.buffer_A[read_offset]),
                0 /* first tile */,
                params.BLOCK_CT_DIM,
                formats.unpack_A_src,
                formats.unpack_A_dst,
                FACE_R_DIM,
                num_faces,
                false /* narrow_tile */,
                DEST_RUN_TILES);
            read_offset += params.BLOCK_CT_DIM;
            continue;
        }
#endif
        _llk_unpack_tilize_block_wrapper_(
            L1_ADDRESS(params.buffer_A[read_offset]),
            0 /* first tile */,
            params.BLOCK_CT_DIM,
            formats.unpack_A_src,
            formats.unpack_A_dst,
            block_ct_dim,
            FACE_R_DIM,
            num_faces,
            false /* narrow_tile */);
#else
        for (std::uint32_t j = 0; j < params.BLOCK_CT_DIM; j++)
        {
            _llk_unpack_tilize_wrapper_(
                L1_ADDRESS(params.buffer_A[read_offset]),
                j,
                formats.unpack_A_src,
                formats.unpack_A_dst,
                block_ct_dim,
                FACE_R_DIM,
                num_faces,
                false /* narrow_tile */);
        }
#endif
        read_offset += params.BLOCK_CT_DIM;
    }
    _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, num_faces);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces = params.num_faces;
    const bool is_int_fpu_en      = false;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // copy srca to dest
    const bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE)
    if (DEST_RUNS && num_faces == 4)
    {
        for (std::uint32_t row = 0; row < params.BLOCK_RT_DIM; ++row)
        {
            for (std::uint32_t tile = 0; tile < params.BLOCK_CT_DIM; tile += DEST_RUN_TILES)
            {
                const std::uint32_t run_tiles = (params.BLOCK_CT_DIM - tile < DEST_RUN_TILES) ? params.BLOCK_CT_DIM - tile : DEST_RUN_TILES;
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_eltwise_unary_datacopy_block_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    0, run_tiles, formats.math, formats.math, num_faces);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        return;
    }
#endif

    const std::uint32_t tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks     = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile = 0; tile < tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math, num_faces);
        }
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
    const std::uint32_t num_faces  = params.num_faces;
    static constexpr bool UNTILIZE = false;

    static constexpr bool TILIZE = true;
    const bool is_8bit_format    = _llk_pack_skip_bh_tilize_workaround_wrapper_(formats.unpack_A_src);
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
    _llk_pack_init_with_src_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats.pack_src,
        formats.pack_dst,
        FACE_R_DIM,
        TILE_C_DIM,
        num_faces,
        false /* partial_face */,
        false /* narrow_tile */,
        1 /* num_tiles */,
        is_8bit_format /* skip_bh_tilize_workaround */);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE)
    if (DEST_RUNS && num_faces == 4)
    {
        for (std::uint32_t row = 0; row < params.BLOCK_RT_DIM; ++row)
        {
            for (std::uint32_t tile = 0; tile < params.BLOCK_CT_DIM; tile += DEST_RUN_TILES)
            {
                const std::uint32_t run_tiles = (params.BLOCK_CT_DIM - tile < DEST_RUN_TILES) ? params.BLOCK_CT_DIM - tile : DEST_RUN_TILES;
                const std::uint32_t first     = row * params.BLOCK_CT_DIM + tile;
                _llk_packer_wait_for_math_done_();
                if (run_tiles > 1 && !IS_BFP_FORMAT(formats.pack_dst))
                {
                    _llk_pack_block_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Tilize>(0, L1_ADDRESS(params.buffer_Res[first]), run_tiles);
                }
                else
                {
                    for (std::uint32_t i = 0; i < run_tiles; ++i)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(i, L1_ADDRESS(params.buffer_Res[first + i]));
                    }
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        return;
    }
#endif

    const std::uint32_t tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks     = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE)
        // One pack run for the block's contiguous tiles
        if (tiles_in_block > 1 && num_faces == 4 && !IS_BFP_FORMAT(formats.pack_dst))
        {
            const std::uint32_t first = block * tiles_in_block;
            if (is_8bit_format)
            {
                _llk_pack_block_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[first]), tiles_in_block);
            }
            else
            {
                _llk_pack_block_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Tilize>(0, L1_ADDRESS(params.buffer_Res[first]), tiles_in_block);
            }
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            continue;
        }
#endif
        for (std::uint32_t tile = 0; tile < tiles_in_block; ++tile)
        {
            std::uint32_t res_tile_idx = (block * tiles_in_block) + tile;
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

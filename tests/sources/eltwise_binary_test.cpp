
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* This test is a generic elementwise binary test, with no broadcast, no transpose
   It can test different tile dimensions, and also different number of tiles/block of tiles*/
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

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    const std::uint32_t face_r_dim = params->TEST_FACE_R_DIM;
    const std::uint32_t num_faces  = params->num_faces_r_dim_A * params->num_faces_c_dim_A;
    const bool narrow_tile         = params->num_faces_c_dim_A < params->num_faces_r_dim_A;
    const std::uint32_t transpose  = params->UNPACK_TRANSPOSE_FACES;
    const int num_tiles_in_block   = params->NUM_TILES_IN_BLOCK;
    const int num_blocks           = params->NUM_BLOCKS;

    // Configure hardware for unpacking, no broadcast, no transpose
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, num_faces, num_faces);

    _llk_unpack_AB_init_<BROADCAST_TYPE>(face_r_dim, num_faces, narrow_tile,
                                         transpose); // Enable face rearrangement for srcA

#ifdef EN_DEST_REUSE
    const int num_total_tiles = params->INPUT_NUM_TILES_IN_BLOCK * params->INPUT_NUM_BLOCKS;
#else
    const int num_total_tiles = params->NUM_TILES_IN_BLOCK * params->NUM_BLOCKS;
#endif

    for (int i = 0; i < num_total_tiles; ++i)
    {
        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params->buffer_A[i]), L1_ADDRESS(params->buffer_B[i]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams *params)
{
    const std::uint32_t num_faces = params->num_faces_r_dim_A * params->num_faces_c_dim_A;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    constexpr bool ACC_TO_DEST = false;
#ifndef EN_DEST_REUSE
    constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::NONE;
#endif

#ifdef EN_DEST_REUSE
    const int tiles_in_block          = params->OUTPUT_NUM_TILES_IN_BLOCK;
    const int num_tiles_accumulations = params->INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
    const int num_blocks              = params->INPUT_NUM_BLOCKS;
#else
    const int tiles_in_block          = params->NUM_TILES_IN_BLOCK;
    const int num_tiles_accumulations = 1;
    const int num_blocks              = params->NUM_BLOCKS;
#endif

    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(num_faces, ACC_TO_DEST);

    for (int block = 0; block < num_blocks; block++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (int n = 0; n < num_tiles_accumulations; n++)
        {
            for (int tile = 0; tile < tiles_in_block; tile++)
            {
                LLK_ASSERT(
                    (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE>(
                    num_faces, tile /* dst_index */, false /* clear_fp32_dst_acc */);
            }
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Cache volatile values to ensure consistent reads
    const std::uint32_t face_r_dim = params->TEST_FACE_R_DIM;
    const std::uint32_t num_faces  = params->num_faces_r_dim_A * params->num_faces_c_dim_A;
    const bool narrow_tile         = params->num_faces_c_dim_A < params->num_faces_r_dim_A;
    const bool partial_face        = face_r_dim < 16;
    const int num_tiles_in_block   = params->NUM_TILES_IN_BLOCK;
    const int num_blocks           = params->NUM_BLOCKS;

    const std::uint32_t tile_size = face_r_dim * 16 * num_faces;

#ifdef ARCH_BLACKHOLE
    const std::uint32_t tile_c_dim = params->num_faces_c_dim_A * 16;
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(
        formats.pack_src, formats.pack_dst, tile_size, face_r_dim, tile_c_dim, num_faces);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(
        formats.pack_src, formats.pack_dst, tile_size, face_r_dim, num_faces, partial_face, narrow_tile);
#endif

#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst, face_r_dim, tile_c_dim, num_faces);
#else
    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst, face_r_dim, num_faces, partial_face, narrow_tile);
#endif

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(face_r_dim, narrow_tile);
#endif

#ifdef EN_DEST_REUSE
    const int output_tiles_in_block = params->OUTPUT_NUM_TILES_IN_BLOCK;
    const int output_num_blocks     = params->OUTPUT_NUM_BLOCKS;
#else
    const int output_tiles_in_block = num_tiles_in_block;
    const int output_num_blocks     = num_blocks;
#endif

    for (int block = 0; block < output_num_blocks; block++)
    {
        _llk_packer_wait_for_math_done_();
        for (int tile = 0; tile < output_tiles_in_block; tile++)
        {
            int res_tile_idx = (block * output_tiles_in_block) + tile;
            LLK_ASSERT(
                (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(tile, L1_ADDRESS(params->buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

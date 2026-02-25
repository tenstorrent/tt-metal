
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
#include "tensor_shape.h"

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
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params->TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params->TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params->num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params->num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const std::uint32_t transpose           = params->UNPACK_TRANSPOSE_FACES;

    // Configure hardware for unpacking, no broadcast, no transpose
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces(),
        tensor_shape.total_num_faces(),
        TILE_SIZE_UNPACK_A,
        TILE_SIZE_UNPACK_B);

    _llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose);

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
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim      = static_cast<std::uint8_t>(params->TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim      = static_cast<std::uint8_t>(params->TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim = static_cast<std::uint8_t>(params->num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim = static_cast<std::uint8_t>(params->num_faces_c_dim_A);
    const TensorShape tensor_shape     = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    constexpr bool ACC_TO_DEST         = false;

    // Initialize math for element-wise operation
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef EN_DEST_REUSE
    const int tiles_in_block          = params->OUTPUT_NUM_TILES_IN_BLOCK;
    const int num_tiles_accumulations = params->INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
    const int num_blocks              = params->INPUT_NUM_BLOCKS;
#else
    const int tiles_in_block          = params->NUM_TILES_IN_BLOCK;
    const int num_tiles_accumulations = 1;
    const int num_blocks              = params->NUM_BLOCKS;
    constexpr auto REUSE_DEST_TYPE    = ckernel::EltwiseBinaryReuseDestType::NONE;
#endif

    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(tensor_shape, ACC_TO_DEST);

    // Perform element-wise operation
    for (int block = 0; block < num_blocks; block++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (int n = 0; n < num_tiles_accumulations; n++)
        {
            for (int tile = 0; tile < tiles_in_block; tile++)
            {
                LLK_ASSERT(
                    (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE>(
                    tensor_shape, tile /* dst_index */, false /* clear_fp32_dst_acc */);
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
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params->TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params->TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params->num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params->num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const int num_tiles_in_block            = params->NUM_TILES_IN_BLOCK;
    const int num_blocks                    = params->NUM_BLOCKS;

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = (tensor_shape.num_faces_c_dim == 1);

#ifdef ARCH_BLACKHOLE
    // BH configure_pack uses partial_face for BFP exp_section_size, but narrow_tile is unused (defaults to false)
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, false /* narrow_tile */);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, num_faces, partial_face, narrow_tile);
#endif

#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces);
#else
    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst, tensor_shape.face_r_dim, num_faces, partial_face, narrow_tile);
#endif

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(tensor_shape.face_r_dim, narrow_tile);
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
                (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(tile, L1_ADDRESS(params->buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

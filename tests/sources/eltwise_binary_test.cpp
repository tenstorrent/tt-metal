
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
    const std::uint32_t num_faces  = params->num_faces;
    const std::uint32_t transpose  = params->UNPACK_TRANSPOSE_FACES;
    const int num_tiles_in_block   = params->NUM_TILES_IN_BLOCK;
    const int num_blocks           = params->NUM_BLOCKS;

    // Configure hardware for unpacking, no broadcast, no transpose
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, face_r_dim, face_r_dim, num_faces, num_faces);

    _llk_unpack_AB_init_<BROADCAST_TYPE>(
        face_r_dim,
        num_faces,
        false,      // narrow_tile
        transpose); // Enable face rearrangement for srcA

    for (int i = 0; i < num_tiles_in_block * num_blocks; ++i)
    {
        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i]));
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
    const std::uint32_t num_faces = params->num_faces;
    const int num_tiles_in_block  = params->NUM_TILES_IN_BLOCK;
    const int num_blocks          = params->NUM_BLOCKS;

    // Initialize math for element-wise subtraction
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWSUB, BROADCAST_TYPE, MathFidelity::LoFi>(num_faces, 0);

    // Perform element-wise subtraction
    for (int block = 0; block < num_blocks; block++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (int tile = 0; tile < num_tiles_in_block; tile++)
        {
            _llk_math_eltwise_binary_<EltwiseBinaryType::ELWSUB, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi>(
                num_faces, tile /* dst_index */, false /* clear_fp32_dst_acc */);
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
    const std::uint32_t num_faces  = params->num_faces;
    const int num_tiles_in_block   = params->NUM_TILES_IN_BLOCK;
    const int num_blocks           = params->NUM_BLOCKS;

    const std::uint32_t tile_size = face_r_dim * 16 * num_faces;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(formats.pack_src, formats.pack_dst, tile_size);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(formats.pack_src, formats.pack_dst, tile_size);
#endif

    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>();
#endif

    for (int block = 0; block < num_blocks; block++)
    {
        _llk_packer_wait_for_math_done_();
        for (int tile = 0; tile < num_tiles_in_block; tile++)
        {
            int res_tile_idx = (block * num_tiles_in_block) + tile;
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(tile, L1_ADDRESS(buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

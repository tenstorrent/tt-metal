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

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        params.num_faces_B, // in1
        params.num_faces_A, // in0
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);
    _llk_unpack_AB_matmul_init_<>(
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM,
        params.KT_DIM,
        params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        params.num_faces_B,     // in1
        params.num_faces_A,     // in0
        params.PARTIAL_FACE_B,  // in1
        params.PARTIAL_FACE_A); // in0
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t j = 0; j < params.KT_DIM; j++)
        {
            _llk_unpack_AB_matmul_<>(
                L1_ADDRESS(params.buffer_A[0]),
                L1_ADDRESS(params.buffer_B[0]),
                j,
                j * params.CT_DIM,
                params.TILE_SIZE_UNPACK_B,
                params.TILE_SIZE_UNPACK_A,
                params.PARTIAL_FACE_B,
                params.PARTIAL_FACE_A,
                params.CT_DIM,
                params.RT_DIM,
                params.KT_DIM);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
        params.in0_tile_r_dim,
        params.in0_tile_c_dim,
        params.in1_tile_r_dim,
        params.in1_tile_c_dim,
        params.PARTIAL_FACE_MATH,
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        LLK_ASSERT(
            (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Matmul block exceeds destination capacity");
        for (std::uint32_t j = 0; j < params.KT_DIM; j++)
        {
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(params.DST_INDEX, params.CT_DIM, params.RT_DIM);
        }

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src,
        formats.pack_dst,
        params.TILE_SIZE_PACK,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces,
        params.PARTIAL_FACE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(params.DST_INDEX + tile, L1_ADDRESS(params.buffer_Res[result_tile]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

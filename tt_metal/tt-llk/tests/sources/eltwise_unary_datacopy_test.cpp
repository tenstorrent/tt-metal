
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

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (!tilize_en)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, FACE_R_DIM, params.num_faces, formats.unpack_A_src, formats.unpack_A_dst);

        const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;

        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    else
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */);

        for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
        {
            const std::uint32_t read_offset = i * BLOCK_CT_DIM;
            for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
            {
                _llk_unpack_tilize_wrapper_(
                    L1_ADDRESS(params.buffer_A[read_offset]),
                    j,
                    formats.unpack_A_src,
                    formats.unpack_A_dst,
                    0 /* block_ct_dim */,
                    FACE_R_DIM,
                    4 /* num_faces */,
                    false /* narrow_tile */);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
// copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, tilize_en, is_int_fpu_en>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile_num, formats.math, formats.math, params.num_faces);
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, false /* untilize */, tilize_en>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<false /* untilize */, false /* zero_output */, tilize_en>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(
                params.DST_INDEX + tile_num, L1_ADDRESS(params.buffer_Res[block_num * params.NUM_TILES_IN_BLOCK + tile_num]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

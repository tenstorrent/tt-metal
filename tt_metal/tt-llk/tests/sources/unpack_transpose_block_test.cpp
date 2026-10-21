// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// transpose_tile's 16-bit and 8-bit path (the unpacker transposes faces and within faces, as transpose_init programs it,
// then the A2D datacopy) with the unpack of each DEST section as one _llk_unpack_A_transpose_block_ call.
// UNPACK_FORM 1: every section is a block. UNPACK_FORM 2: blocks and per tile calls alternate by section.
// UNPACK_FORM 3 and 4: forms 1 and 2 with the back to back block, whose W counter walks through the tiles.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE, false>(
        1, 1, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces), formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t tile_stride = L1_ADDRESS(params.buffer_A[1]) - L1_ADDRESS(params.buffer_A[0]);
    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        const std::uint32_t first = block * num_tiles_in_block;
        if (UNPACK_FORM % 2 == 1 || block % 2 == 0)
        {
            _llk_unpack_A_transpose_block_<(UNPACK_FORM >= 3)>(L1_ADDRESS(params.buffer_A[first]), num_tiles_in_block, tile_stride);
        }
        else
        {
            for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                    L1_ADDRESS(params.buffer_A[first + tile]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    if (masked_data_format(formats.math) == to_underlying(DataFormat::Int8))
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            true /* is_int_fpu_en */,
            PackMode::Default>(params.num_faces, formats.math);
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            params.num_faces, formats.math);
    }
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                tile, formats.math, formats.math);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile, L1_ADDRESS(params.buffer_Res[(block * num_tiles_in_block) + tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

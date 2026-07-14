
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        params.UNPACK_TRANSPOSE_FACES,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // Feed SrcA (for the MATH datacopy) then dummy-valid SrcB (for the transpose)
    // one DEST bank (block) at a time, matching the MATH thread's per-block
    // datacopy+transpose. Doing all SrcA feeds up front would withhold a block's
    // SrcB valids behind later blocks' SrcA feeds and stall the transpose.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[block * params.NUM_TILES_IN_BLOCK + tile]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_unpack_set_srcb_dummy_valid_();
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
    constexpr bool is32 = is_fp32_dest_acc_en;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Process the input one DEST bank (block) at a time so inputs with more
    // tiles than fit in DEST exercise DEST bank switching. Each block datacopies
    // its tiles into DEST, transposes them in place, then hands the bank to the
    // packer via the wait/section_done pair.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

        // A2D datacopy and transpose_dest need different ALU data-format state,
        // so (re)configure datacopy at the start of each block's copy phase.
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
            params.num_faces, formats.math);
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }

        _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is32>();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, MATH_TRANSPOSE_FACES, is32>(tile);
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Pack one DEST bank (block) at a time, mirroring the MATH thread's bank
    // switching; the block-local DEST tile index maps to the global result tile.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile, L1_ADDRESS(params.buffer_Res[block * params.NUM_TILES_IN_BLOCK + tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

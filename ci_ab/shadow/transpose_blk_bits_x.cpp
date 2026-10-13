// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk_analysis transpose family headroom bit dump kernel (Blackhole), not part of the tree. transpose_tile of 16-bit and
// 8-bit tensors as transpose_init programs the unpacker (faces and within face, acc_to_dest = true), the A2D datacopy
// (integer FPU path for Int8) and the pack, SECTION_TILES tiles per DEST section.
// UNPACK_FORM 0: one _llk_unpack_A_ per tile (main). 1: one _llk_unpack_A_transpose_block_ per section.
// 4: block and per tile calls alternate by section, so each form runs after the other.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

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
    _llk_unpack_A_init_<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE, false>(
        1, 1, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces), formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t stride = L1_ADDRESS(params.buffer_A[1]) - L1_ADDRESS(params.buffer_A[0]);
    for (std::uint32_t section = 0; section < params.TILE_CNT; section += SECTION_TILES)
    {
        const bool block = UNPACK_FORM == 1 || (UNPACK_FORM == 4 && (section / SECTION_TILES) % 2 == 0);
        if (block)
        {
            if (IS_BFP_FORMAT(formats.unpack_A_src))
            {
                _llk_unpack_A_transpose_block_(L1_ADDRESS(params.buffer_A[section]), SECTION_TILES, stride);
            }
            else
            {
                _llk_unpack_A_transpose_block_<true>(L1_ADDRESS(params.buffer_A[section]), SECTION_TILES, stride);
            }
        }
        else
        {
            for (std::uint32_t tile = 0; tile < SECTION_TILES; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                    L1_ADDRESS(params.buffer_A[section + tile]), formats.unpack_A_src, formats.unpack_A_dst);
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
    const bool is_int8 = (masked_data_format(formats.math) == to_underlying(DataFormat::Int8));
    if (is_int8)
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, true, PackMode::Default>(
            params.num_faces, formats.math);
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            params.num_faces, formats.math);
    }
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (std::uint32_t section = 0; section < params.TILE_CNT; section += SECTION_TILES)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile = 0; tile < SECTION_TILES; ++tile)
        {
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (std::uint32_t section = 0; section < params.TILE_CNT; section += SECTION_TILES)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < SECTION_TILES; ++tile)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[section + tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

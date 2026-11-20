// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// max_pool_with_indices (Wormhole): column-wise arg-max of a values tile, carrying an indices tile in lockstep.
//
// buffer_A[i] -> Dest[i] for i < TILE_CNT. SRC0_TILE_IDX is the values tile, SRC1_TILE_IDX the indices tile;
// with MAX_POOL_ACCUMULATE the running max lives in Dest[SRC0 + 1] / Dest[SRC1 + 1]. PACK writes
// Dest[DST_TILE_IDX] (one tile). The kernel is called the way compute_mpwi.cpp calls it: once per tile pair,
// under VectorMode::None.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
    const Operand& buffer_A      = params.buffer_A;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_sfpu/ckernel_sfpu_max_pool_indices.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT       = params.TILE_CNT;
    const std::uint32_t SRC0_TILE_IDX  = params.SRC0_TILE_IDX;
    const std::uint32_t SRC1_TILE_IDX  = params.SRC1_TILE_IDX;
    const std::uint32_t MAX_POOL_CHUNK = params.MAX_POOL_CHUNK;
#endif
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
    SFPU_BINARY_INIT_FN(max_pool_with_indices, sfpu::init_max_pool_with_indices, (APPROX_MODE, MAX_POOL_LAYOUT));

    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }
    SFPU_BINARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        calculate_max_pool_with_indices,
        (APPROX_MODE, is_fp32_dest_acc_en, MAX_POOL_NUM_ROWS, 8 /* ITERATIONS */, MAX_POOL_LAYOUT, MAX_POOL_ACCUMULATE),
        SRC0_TILE_IDX,
        SRC1_TILE_IDX,
        0 /* unused out tile */,
        VectorMode::None,
        MAX_POOL_CHUNK);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t DST_TILE_IDX = params.DST_TILE_IDX;
    const Operand& buffer_Res        = params.buffer_Res;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_TILE_IDX, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK

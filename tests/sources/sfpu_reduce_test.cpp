// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);

    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params->buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(const volatile struct RuntimeParams* params)
{
// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(4, formats.math);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t i = 0; i < params->TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math);
    }

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();

    ckernel::sfpu::_init_reduce_<POOL_TYPE, static_cast<DataFormat>(formats.math)>();

    if (REDUCE_DIM == ReduceDim::REDUCE_COL)
    {
        for (std::uint32_t i = 0; i < params->TILE_CNT; ++i)
        {
            LLK_ASSERT(
                (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(i);
            ckernel::sfpu::_calculate_reduce_<POOL_TYPE, REDUCE_DIM, static_cast<DataFormat>(formats.math)>();
        }
    }
    else if (REDUCE_DIM == ReduceDim::REDUCE_ROW)
    {
        _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);
        ckernel::sfpu::_calculate_reduce_<POOL_TYPE, REDUCE_DIM, static_cast<DataFormat>(formats.math)>(BLOCK_CT_DIM, BLOCK_RT_DIM);
    }

#ifdef ADD_TOP_ROW
    _llk_math_eltwise_binary_sfpu_init_<SfpuType::add_top_row>();
    _llk_math_eltwise_binary_sfpu_start_<DstSync::SyncHalf>(0);
    ckernel::sfpu::_init_add_top_row_();

    for (int i = 1; i < params->TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        // Add the top rows of all the tiles we reduced in dst register
        ckernel::sfpu::_calculate_add_top_row_<static_cast<DataFormat>(formats.math)>(0, i, 0); // accumulate the result in tile at index 0
    }

#endif

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, false>();
#endif

    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(params->buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

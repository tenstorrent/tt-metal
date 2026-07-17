// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

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
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "params.h"
#include "sfpu_operations.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const bool is_int_fpu_en         = false;
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(i, formats.math, formats.math);
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();

    test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, 32 /* iterations */, formats.math>();

    test_utils::call_binary_sfpu_operation<DstSync::SyncHalf, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, 32 /* iterations */, formats.math>(
        0 /* dst_index_in0 */, 1 /* dst_index_in1 */, 0 /* dst_index_out */);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for `add_int<APPROX, ITERATIONS>(dst_offset)` of llk_sfpu/ckernel_sfpu_int_sum.h (the kernel
// behind the compute API's sfpu_add_int(idst, dst_offset)): Dest tile idst += Dest tile idst + dst_offset,
// in place. TILE_CNT Int32 tiles are copied into Dest tiles 0..TILE_CNT-1, the add runs once with the
// compile-time INT_SUM_DST_INDEX and INT_SUM_DST_OFFSET, and every Dest tile is packed back so the test
// can check both the sum and that the other tiles were left alone.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Every input tile lands in the Dest tile of the same index, all within one Dest half.
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile /*dst_index*/, formats.math, formats.math);
    }

    // The compute API's sfpu_sum_int_init() followed by sfpu_add_int(dst_index, INT_SUM_DST_OFFSET), including
    // its bound on the operand tile (SFPU_UNARY_CALL only checks the destination index).
    constexpr std::uint32_t dst_index = INT_SUM_DST_INDEX;
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::sum_int_init<APPROX_MODE>);
    LLK_ASSERT(
        (static_cast<std::uint64_t>(dst_index) + INT_SUM_DST_OFFSET < get_dest_max_tiles_rt<DST_SYNC, DstTileShape::Tile32x32>()),
        "sfpu_add_int operand tile idst + dst_offset exceeds max dest tiles");
    SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, add_int, (APPROX_MODE, ITERATIONS), dst_index, VECTOR_MODE, INT_SUM_DST_OFFSET);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /*tile_size*/);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile /*tile_index*/, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

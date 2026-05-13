// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
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

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, TILE_NUM_FACES, formats.unpack_A_src, formats.unpack_A_dst);

    for (int i = 0; i < params.NUM_BLOCKS; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

const int iterations = 32;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // copy srca to dest — use the cross-arch wrapper (same as eltwise_unary_sfpu_test.cpp)
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    test_utils::call_unary_sfpu_operation_init<
        SFPU_UNARY_OPERATION,
        APPROX_MODE,
        is_fp32_dest_acc_en,
        iterations,
        FAST_MODE,
        false /* STABLE_SORT */,
        CLAMP_NEGATIVE>();

    LLK_ASSERT((DST_INDEX_IN < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "DST_INDEX_IN exceeds max dest tiles");
    LLK_ASSERT((DST_INDEX_OUT < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "DST_INDEX_OUT exceeds max dest tiles");

    for (int block = 0; block < params.NUM_BLOCKS; block++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        // Data-copy input tile into DEST at position DST_INDEX_IN
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            DST_INDEX_IN, formats.math, formats.math);

        // Run SFPU: read from DST_INDEX_IN, write to DST_INDEX_OUT. The split
        // helper routes through ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>
        // (dst-bound LLK_ASSERT) and then _llk_math_eltwise_unary_sfpu_params_split_,
        // which calls the op's calculate function with
        // (dst_index_in, dst_index_out, args...).
        test_utils::call_unary_sfpu_operation_split<
            DST_SYNC,
            is_fp32_dest_acc_en,
            SFPU_UNARY_OPERATION,
            APPROX_MODE,
            is_fp32_dest_acc_en,
            iterations,
            FAST_MODE,
            false /* STABLE_SORT */,
            CLAMP_NEGATIVE>(DST_INDEX_IN, DST_INDEX_OUT, formats.math);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<false /* untilize */, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (int block = 0; block < params.NUM_BLOCKS; block++)
    {
        _llk_packer_wait_for_math_done_();

        // Pack from DST_INDEX_OUT — the SFPU wrote its result here
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, /* untilize */ false>(DST_INDEX_OUT, L1_ADDRESS(params.buffer_Res[block]));

        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

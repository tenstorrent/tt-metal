// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone LLK test for the SFPU-backed CB hash (23-bit "FNV23"). See
// tt_llk_{blackhole,wormhole_b0}/llk_lib/debug/llk_math_hash_cb.h.
//
// Data path:
//   UNPACK: unpacks TILE_CNT INT32 tiles from buffer_A into DEST slot 0.
//   MATH:   datacopies each tile into DEST, folds it into 32 per-lane FNV23
//           accumulators, then writes the accumulators back into DEST (row 0;
//           the rest of the tile is zeroed).
//   PACK:   packs the DEST tile to buffer_Res[0].
//
// The host XOR-folds the whole result tile; the zeroed rows cancel, leaving
// XOR(32 accumulators). Uses only the proven INT32 DEST -> PACK -> L1 path.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define DEBUG_CB_HASH 1

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
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "debug/llk_math_hash_cb.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool is_int_fpu_en = true;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    _llk_math_hash_cb_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        // Datacopy A → DEST slot 0 so the SFPU can read it, then fold it in.
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0, formats.math, formats.math);
        _llk_math_hash_cb_tile_(/*dst_tile_idx=*/0);
    }

    // Write the per-lane accumulators back into DEST for the packer.
    _llk_math_hash_cb_store_to_dest_();

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Raw INT32 bitcast pack (pack_src == pack_dst == Int32): the packer copies
    // DEST datums without a numeric (FP32) conversion.
    constexpr std::uint32_t int32_fmt = static_cast<std::uint32_t>(DataFormat::Int32);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        int32_fmt, int32_fmt, 32 * 32 * 4 /* INT32 tile size in bytes */, FACE_R_DIM, 4 /* num_faces */);
    // The hw-configure above established the packer strides, so the wrapper skips re-programming them;
    // it still programs the X (datum) counter, which configure_pack does not touch.
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(int32_fmt, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

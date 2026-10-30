// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* UNPACK moves one tile from L1 into DEST (unary datacopy, unpack-to-dest
   for 32-bit formats), then MATH reads that tile out of DEST through the
   RISC-V debug window (dbg_copy_dest_tile<Read>) into the result buffer.
   Unlike the write->read round trip in debug_dest_copy.cpp, the producer
   here is the unpacker, so a conversion applied by the debug window on the
   way out is not cancelled by a matching one on the way in. */

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

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
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /*transpose_of_faces*/,
        0 /*within_face_16x16_transpose*/,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    const DataFormat l1_fmt          = static_cast<DataFormat>(formats.unpack_A_src);
    constexpr std::uint32_t TILE_IDX = 0;

    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        false /*is_int_fpu_en*/,
        llk_test_pack_mode_v<false /*untilize*/, false /*tilize*/>>(TILE_NUM_FACES, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        TILE_IDX, formats.math, formats.math, TILE_NUM_FACES);

    // The tile is now in DEST; read it back through the debug window.
    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, TILE_IDX, reinterpret_cast<void*>(params.buffer_Res[0]));

    // No _llk_math_dest_section_done_: PACK is idle here, so nothing would consume the
    // MATH_PACK post, and the next kernel's _llk_math_pack_sync_init_ would wait on it forever.
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packing a 32-bit integer output after a mid-kernel dest-acc enable / disable pair.
//
// Built with is_fp32_dest_acc_en = false and an Int32 output, so the packer's full configure selects
// the 32-bit Dest read for the Int32 pack source. MATH enables FP32 dest-acc, A2D-copies the Float16_b
// tile into the 32-bit Dest, converts it in place to Int32 with the production typecast primitive, and
// disables dest-acc again. PACK then packs the Int32 tile. The disable must keep the packer's 32-bit
// Dest read: the pack source format needs it whatever the accumulation mode, exactly as the packer's
// own configure derives it. The Python driver compares the packed words with the converted stimuli.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_fp32_dest_acc.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;
constexpr std::uint32_t TILE_IDX           = 0;

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
        0 /*transpose_of_faces*/,
        0 /*within_face_16x16_transpose*/,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // UNPACK only takes part in the two handshakes: enable, then disable.
    _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"

using namespace ckernel;

// Load / store pairs per face: each moves 32 datums, a 16x16 face holds 256.
constexpr int iterations = 8;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // 32-bit Dest for the section that produces the Int32 tile.
    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true /*enable*/);

    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        true /*is_fp32_dest_acc_en*/,
        BroadcastType::NONE,
        false /*is_int_fpu_en*/,
        PackMode::Default>(TILE_NUM_FACES, formats.math);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, true /*is_fp32_dest_acc_en*/, BroadcastType::NONE, unpack_to_dest>(
        TILE_IDX, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_typecast_fp32_to_int32<false /*APPROXIMATION_MODE*/, iterations>, TILE_IDX, VectorMode::RC);

    // Back to the kernel's 16-bit mode before PACK reads the tile.
    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false /*enable*/);

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
    constexpr std::uint32_t tile_size = FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // PACK takes part in the two handshakes, enable then disable, reporting the Int32 pack source.
    _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>(static_cast<DataFormat>(formats.pack_src));
    _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>(static_cast<DataFormat>(formats.pack_src));

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(TILE_IDX, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

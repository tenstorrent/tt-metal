// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// binop_with_scalar is float-only (add/sub/mul/div/rsub). All tested formats
// (Float32, Float16_b, Bfp8_b) pass through untouched; anything unexpected
// falls back to a raw 16-bit (UInt16) move so the harness still links.
static constexpr std::uint8_t resolve_binop_format(std::uint8_t in)
{
    switch (static_cast<DataFormat>(in))
    {
        case DataFormat::Float32:
        case DataFormat::Bfp8_b:
        case DataFormat::Float16_b:
            return in;
        default:
            return static_cast<std::uint8_t>(DataFormat::UInt16);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t UNPACK_FMT = resolve_binop_format(UNPACK_A_IN);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, FACE_R_DIM, 4 /* num_faces */, UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), UNPACK_FMT, UNPACK_FMT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

// calculate_binop_with_scalar() below reads DST_ACCUM_MODE: its RSUB branch applies
// the fp32->bf16 RNE correction only for the 16-bit-dest (dest_acc:No) path, exactly
// like production. It's a real constexpr, not the sfpu_operations.h #define hack.
static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;

#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint8_t MATH_FMT = resolve_binop_format(UNPACK_A_IN);
    const bool is_int_fpu_en    = false;

    // Copy the single input tile into Dest tile 0, then run the scalar binop in place.
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, MATH_FMT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, MATH_FMT, MATH_FMT);
    // Reset dest addressing before the sfpi-based op so the dest RWC is at the
    // tile-0 base (matches the ternary/binary SFPU tests).
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // Scalar binop: out(tile 0) = binop(dst, scalar). VectorMode::RC drives 4
    // faces (8 rows each), so ITERATIONS is 8 per call, matching the production
    // add_unary_tile / sub_unary_tile / ... APIs.
    SFPU_UNARY_INIT(unused);
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX_MODE, SFPU_BINOP_MODE, 8 /* ITERATIONS */),
        0 /* dst_index */,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = resolve_binop_format(UNPACK_A_IN);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

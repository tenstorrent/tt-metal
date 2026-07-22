// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the fused o_norm SFPU entry (ckernel_sfpu_o_norm.h via
// llk_math_o_norm_sfpu_entry.h):
//
//     o_norm = RMSNorm(o) * gamma2 * sigmoid(g_out)
//
// The three operands (o, gamma2, g_out) each span NUM_REDUCE_TILES tiles and
// are laid out in consecutive Dest tile ranges; the RMSNorm reduction runs down
// the rows (per column / head). See ckernel_sfpu_o_norm.h for the tilization
// contract. The result is written to its own Dest tile range and packed out.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// Dest tile ranges (each operand occupies NUM_REDUCE_TILES consecutive tiles).
static constexpr std::uint32_t O_NORM_DST_O     = 0;
static constexpr std::uint32_t O_NORM_DST_GAMMA = NUM_REDUCE_TILES;
static constexpr std::uint32_t O_NORM_DST_GOUT  = 2 * NUM_REDUCE_TILES;
static constexpr std::uint32_t O_NORM_DST_OUT   = 3 * NUM_REDUCE_TILES;

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
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // Unpack o, then gamma2, then g_out, so the datacopy loop lands them in
    // consecutive Dest tile ranges.
    for (std::uint32_t i = 0; i < NUM_REDUCE_TILES; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
    for (std::uint32_t i = 0; i < NUM_REDUCE_TILES; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_B[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
    for (std::uint32_t i = 0; i < NUM_REDUCE_TILES; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_C[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_o_norm_sfpu_entry.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // All operands must be resident in Dest simultaneously for the reduction.
    for (std::uint32_t tile = 0; tile < 3 * NUM_REDUCE_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    // Reset the dest RWC to the tile-0 base after the datacopies advanced it, so
    // the sfpi dst_reg[...] absolute offsets in calculate_o_norm start from tile
    // 0 (matches sfpu_ternary_test.cpp). Init the SFPU only after this reset.
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();
    llk_math_o_norm_sfpu_init<APPROX_MODE>();

    llk_math_o_norm_sfpu<APPROX_MODE, is_fp32_dest_acc_en, static_cast<DataFormat>(UNPACK_A_IN), NUM_REDUCE_TILES>(
        O_NORM_DST_O, O_NORM_DST_GAMMA, O_NORM_DST_GOUT, O_NORM_DST_OUT, O_NORM_EPS_BITS);

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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < NUM_REDUCE_TILES; ++i)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(O_NORM_DST_OUT + i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercises llk::sfpu::eltwise_unary / eltwise_binary / eltwise_ternary (sfpu/eltwise.h).
//
// The tile is always a full 32x32 (four faces) through unpack and pack. Only the shape
// eltwise walks varies, so the packed result shows exactly which faces unary
// touched:
//
//   num_faces 2x2 -> faces 0,1,2,3   (what VectorMode::RC did)
//   num_faces 1x2 -> faces 0,1       (VectorMode::R)
//   num_faces 2x1 -> faces 0,2       (VectorMode::C -- the skipped-face advance)
//
// The op is negation, so a face-selection error shows up as a sign flip -- far outside the
// format tolerance the result is compared at. The comparison is tolerance-based, not exact:
// the L1 -> DEST -> L1 round trip is lossy whenever the dest is narrower than the L1 format.

#include <cstdint>

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
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/eltwise.h"

using namespace ckernel;

// The region unary should touch. The tile itself stays full, so faces outside this shape, and
// rows past the last 4-row group the walk covers, must come back unmodified.
constexpr ckernel::TensorShape ELTWISE_SHAPE = ckernel::make_tensor_shape(WALK_FACE_R_DIM, FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_once_();

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            // L1 -> SrcA -> DEST, so unary has a full tile to work on in place.
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math);

            // In place: the same tile is every operand. Negation does not combine lanes, so the walk may run
            // past a short face's rows. Each functor reduces to -a, so the golden does not depend on the arity.
            constexpr auto AGNOSTIC = llk::sfpu::PaddingAgnostic::Yes;
            if constexpr (ELTWISE_ARITY == 1)
            {
                llk::sfpu::eltwise_unary<ELTWISE_SHAPE, AGNOSTIC>(block_tile, block_tile, [](sfpi::vFloat a) { return -a; });
            }
            else if constexpr (ELTWISE_ARITY == 2)
            {
                llk::sfpu::eltwise_binary<ELTWISE_SHAPE, AGNOSTIC>(
                    block_tile, block_tile, block_tile, [](sfpi::vFloat a, sfpi::vFloat b) { return a - b - b; });
            }
            else
            {
                llk::sfpu::eltwise_ternary<ELTWISE_SHAPE, AGNOSTIC>(
                    block_tile, block_tile, block_tile, block_tile, [](sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) { return a - b - c; });
            }
        }

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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                block_tile, L1_ADDRESS(params.buffer_Res[block_start * params.NUM_TILES_IN_BLOCK + block_tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

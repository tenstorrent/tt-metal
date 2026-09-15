// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// copy_dest_value (DEST tile -> DEST tile) with the destination slot ALREADY
// OCCUPIED by a preceding op.
//
// The MATH section below is a literal expansion of the public compute API
// ``copy_dest_values<DATA_FORMAT>(idst_in, idst_out)``
// (api/compute/copy_dest_values.h): same SFPU_BINARY_INIT_FN_NO_ARGS(unused, ...)
// init, same SFPU_BINARY_CALL with (DATA_FORMAT, false) and VectorMode::RC. Only
// the surrounding prelude is hand-written, because tt-llk kernels drive the
// _llk_* layer directly.
//
// WHAT THIS ADDS OVER test_dest_copy.py::test_dump_dest
// ------------------------------------------------------
// test_dump_dest exercises an L1 -> DEST -> L1 round trip through the RISC-V
// debug window with nothing else resident in DEST. The interesting case for
// copy_dest_value is the opposite one: a real op has just written BOTH the
// source and the destination DEST slot, and the copy has to overwrite the
// destination's existing contents. That is how the primitive is used in
// practice -- stage a value you still need to pack, then transform the copy --
// and it is the case no existing test covers.
//
// SHAPE
// -----
//   1. datacopy TILE_CNT tiles from L1 into DEST[0 .. TILE_CNT-1]
//   2. copy_dest_value(SRC0_TILE_IDX -> SRC1_TILE_IDX)
//   3. pack DEST[0] and DEST[1] out to Res[0], Res[1]
//
// The host checks two things per variant:
//   * the SOURCE slot is unchanged (self-check: proves the prelude, the tile
//     addressing and the pack all target what the test thinks they do, so a
//     kernel-authoring mistake cannot masquerade as a copy defect);
//   * the DESTINATION slot now holds the source tile's data (the actual claim).
//
// The tile geometry is a parameter, which is the point of the tiny-tile cells:
// copy_dest_value hardcodes the FULL-tile DST stride in two places --
// _llk_math_eltwise_sfpu_start_ uses set_dst_write_addr<DstTileShape::Tile32x32>
// (tile_index << 6, i.e. x64 rows) and the functor uses its own
// dst_tile_size = 64 -- while DstTileSizeLog2 is 5 for Tile32x16 and 4 for
// Tile16x16. VectorMode::RC also walks a fixed four faces regardless of how
// many the tile has. So for a tile that is not 32x32 the primitive may address
// a different slot than the surrounding op and the packer do.
//
// TILE_CNT selects the occupancy: TILE_CNT=2 means the destination slot holds a
// different tile's data when the copy runs, TILE_CNT=1 means it was never
// written. Both copy directions (0 -> 1 and 1 -> 0) are covered, which
// separates "the store never lands" from "the store lands at the load's
// offset".

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Tiles packed back to L1: DEST[0] and DEST[1], so either copy direction can be
// graded without the host having to know which slot moved.
static constexpr std::uint32_t RESULT_TILES = 2;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces(),
        tensor_shape.total_num_faces());
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }

    _llk_unpack_A_uninit_<BroadcastType::NONE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu_copy_dest_values.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // A metal kernel gets this from compute_kernel_hw_startup; this standalone
    // harness bypasses it, so run the idempotent once-init before any SFPU store.
    _llk_math_eltwise_unary_sfpu_init_once_();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // The preceding op: a plain datacopy, so every DEST slot it touches holds a
    // known tile and any deviation in the result is attributable.
    const std::uint32_t num_faces = static_cast<std::uint32_t>(params.num_faces_r_dim_A) * static_cast<std::uint32_t>(params.num_faces_c_dim_A);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        num_faces, formats.math);
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
            tile, formats.math, formats.math, num_faces);
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, false /* unpack_to_dest */>();

    // ---- copy_dest_values<COPY_DEST_VALUES_FORMAT>(SRC0_TILE_IDX, SRC1_TILE_IDX) ----
    // Verbatim from api/compute/copy_dest_values.h, minus its MATH() wrapper.
    // The functor signature is copy_dest_value(dst_index_in, dst_index_out,
    // unused), so DST_IN0 is the source slot, DST_IN1 the destination slot, and
    // DST_OUT is the unused third argument.
    const std::uint32_t copy_src_slot = params.SRC0_TILE_IDX;
    const std::uint32_t copy_dst_slot = params.SRC1_TILE_IDX;
    const std::uint32_t copy_unused   = params.DST_TILE_IDX;

    SFPU_BINARY_INIT_FN_NO_ARGS(unused, ckernel::sfpu::copy_dest_value_init);
    SFPU_BINARY_CALL(
        DST_SYNC,
        is_fp32_dest_acc_en,
        copy_dest_value,
        (COPY_DEST_VALUES_FORMAT, false /* APPROXIMATE */),
        copy_src_slot,
        copy_dst_slot,
        copy_unused,
        VECTOR_MODE);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tensor_shape.total_tensor_size(), tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);
    _llk_pack_init_<PackMode::Default, false /* zero_output */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, RESULT_TILES, false /* skip_bh_tilize_workaround */);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < RESULT_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

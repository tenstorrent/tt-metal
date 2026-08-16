// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the SFPU RoPE
// (tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_rope.h).
//
// sfpu_rope_all_rows is not an eltwise-unary SFPU kernel: it addresses DEST itself in
// absolute rows and does its own dest setup, so there is no SFPU_UNARY_CALL and no
// VectorMode here. The call sequence mirrors what the compute API
// (hw/inc/api/compute/experimental/rope_sfpu.h) emits: configure the addrmod once at
// init, then dest_setup + all_rows inside the dest section.
//
// All TILE_CNT tiles are datacopy'd into ONE dest section, tile i at DEST row 64*i,
// and every tile is packed back out. The rotation only covers the rows the ROPE_*
// operand addresses point at, so packing everything lets the test assert that the
// cos/sin operands and the rest of DEST came back untouched.
//
// The operand addresses being absolute DEST rows is what lets the test drive both
// layouts the LLK documents: stride 64 for copy_tile-shaped operands, and stride 32
// for a dense-packed matmul result, where two consecutive operands share one tile
// slot's faces.
//
// One SFPU vector is 4 DEST rows x the 16 columns of one face, and the LLK issues one
// per (width tile, face), so each x operand has only rows base..base+3 of each of its
// two faces rotated.
//
// The loads and stores are hardcoded to InstrModLoadStore::FP16B, so this op is
// bf16-DEST only: dest_acc=Yes would reinterpret 32-bit DEST words as bf16. The test
// sweeps dest_acc=No alone.

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

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_rope.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(ROPE_WT == 1 || ROPE_WT == 2, "rope: Wt must be 1 or 2 (decode rotary head_dim <= 64)");

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // Leaves ADDR_MOD_7 as {srca:0, srcb:0, dest:0}, which is what the rope's own
    // addrmod configuration also programs.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    sfpu::sfpu_rope_configure_addrmod();

    // One dest section holds every operand: the rotation reads cos/sin and the x tiles
    // out of DEST at the same time, so they cannot be streamed a tile at a time.
    _llk_math_wait_for_dest_available_<DST_SYNC>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    sfpu::sfpu_rope_dest_setup();
    sfpu::sfpu_rope_all_rows<ROPE_HT, ROPE_WT, ROPE_X_BASE, ROPE_X_STRIDE, ROPE_COS_BASE, ROPE_SIN_BASE, ROPE_CS_STRIDE, ROPE_HAS_SCALE>(ROPE_SCALE_FP32);

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
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole-only unit test for the experimental SDPA custom matmul LLK
// (api/compute/experimental/sdpa_custom_mm.h -> sdpa_custom_mm_block, promoted by
// tt-metal #53295, item 3).
//
// This test replicates the body of ckernel::sdpa_custom_mm_block(...) call-for-call from
// the low-level LLKs, because a tt-llk test cannot include tt_metal/hw/inc/api/compute.
// The three threads mirror the compute API exactly:
//
//   UNPACK  llk_unpack_AB_custom_mm_init<transpose>   -> _llk_unpack_AB_custom_mm_init_
//           llk_unpack_AB_sdpa_custom_mm<read_transposed>
//                                                      -> _llk_unpack_AB_sdpa_custom_mm_
//   MATH    llk_math_sdpa_custom_mm_init<transpose>    -> _llk_math_sdpa_custom_mm_init_
//           llk_math_sdpa_custom_mm<signal_granularity>-> _llk_math_sdpa_custom_mm_
//   PACK    llk_pack_* + the sdpa_custom_mm_block_init_pack_short() Z/W-stride RMWs
//
// ---------------------------------------------------------------------------------------
// GOLDEN DERIVATION (mirrors llk_math_sdpa_custom_mm.h + the block() wrapper)
// ---------------------------------------------------------------------------------------
// sdpa_custom_mm is the SDPA QK^T / PV matmul. Operand mapping (from the unpack API,
// llk_unpack_AB_sdpa_custom_mm_api.h): operand0 (in0) -> SrcB, operand1 (in1) -> SrcA.
// In Tensix matmul the lhs is unpacked into SrcB and the rhs into SrcA, so:
//
//     out[M,N] = in0[M,K] @ in1[K,N]          (LoFi fidelity, the only mode this LLK supports)
//
// with the custom_mm layout limits (llk_unpack_AB_custom_mm.h):
//     in0 (SrcB) tile shape [M, 32], M in {1,2,4,8}   -> operandB_face_r_dim
//     in1 (SrcA) tile shape [32, 32]
//     rt_dim = 1, ct_dim = N/32 in [1,16], kt_dim = K/32 even in [2,256]
//
// _llk_math_sdpa_custom_mm_ does exactly one accumulating MVMUL walk:
//   - _llk_math_sdpa_custom_mm_mask_dest_ first ZEROes the ct_dim result tiles in DEST
//     (mask_chunk=false path: TT_ZEROACC per tile). mask_chunk=true would instead unpack a
//     mask tile into SrcB and MOVB2D-broadcast it; that path needs an SFPU-produced mask CB
//     and is NOT exercised here (see the .py). With mask_chunk=false the DEST simply starts
//     at zero, so the result is the plain accumulated product below.
//   - It then runs the kt-deep MVMUL MOP (kt_dim/2 iterations, each covering 2 k-tiles),
//     accumulating in0[:, k] * in1[k, :] over all k into the ct_dim output tiles.
//   - The FPU->SFPU semaphore posts (signal_granularity cadence) are pure signalling to a
//     downstream SFPU consumer; they do NOT change the numeric result. With no SFPU thread
//     in this test the posts just increment an unread semaphore. We test the default
//     signal_granularity = 1 (post per c-tile).
//
// So the golden is the standard LoFi tiled matmul A@B, computed on the host with
// MatmulGolden (LoFi), and only the M defined output rows per tile are validated -- the
// remaining 32-M rows of each output tile are undefined padding (in0 only has M rows).
//
// signal_granularity == ct_dim takes the single-post fast path in the header; the numeric
// result is identical, so it is a compile-only variant here.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// SIGNAL_GRANULARITY is emitted by the SDPA_SIGNAL_GRANULARITY template param.
// READ_TRANSPOSED / MM_TRANSPOSE are emitted by SDPA_CUSTOM_MM_FLAGS.
#ifndef SIGNAL_GRANULARITY
#define SIGNAL_GRANULARITY 1
#endif
#ifndef READ_TRANSPOSED
#define READ_TRANSPOSED false
#endif
#ifndef MM_TRANSPOSE
#define MM_TRANSPOSE false
#endif

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // in1 -> SrcA (rhs, [32,32] tiles), in0 -> SrcB (lhs, [M,32] narrow tiles).
    // _llk_unpack_hw_configure_ takes (unpA, unpB) pairs; unpA = SrcA = in1, unpB = SrcB = in0.
    // Both inputs are Float16_b, so the src/dst formats are identical either way; the face
    // geometry is what differs (in1: 16-row faces x4, in0: M-row faces x2).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, // unpA_src : in1 (SrcA)
        formats.unpack_B_src, // unpB_src : in0 (SrcB)
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.in1_face_r_dim,      // unpA_face_r_dim : in1 = FACE_R_DIM (16)
        params.in0_face_r_dim,      // unpB_face_r_dim : in0 = M
        params.num_faces_A,         // unpA_num_faces  : in1 = 4
        params.num_faces_B,         // unpB_num_faces  : in0 = 2
        params.TILE_SIZE_UNPACK_A,  // unpA_tile_size  : in1 [32,32]
        params.TILE_SIZE_UNPACK_B); // unpB_tile_size  : in0 [M,32]

    // init: unpB_face_r_dim = in0 row count (M), unpA_dst_format selects the profiling
    // heuristic (post1) in the MOP config.
    _llk_unpack_AB_custom_mm_init_<MM_TRANSPOSE>(params.in0_face_r_dim, formats.unpack_A_dst, CT_DIM);

    // Run: base_address_a = in1 (SrcA), base_address_b = in0 (SrcB), mask address 0 (no mask).
    // The whole kt walk is issued by a single call (internal MOP covers kt_dim/2 iterations).
    _llk_unpack_AB_sdpa_custom_mm_<READ_TRANSPOSED>(
        L1_ADDRESS(params.buffer_A[0]), // base_address_a : in1 (SrcA, rhs)
        L1_ADDRESS(params.buffer_B[0]), // base_address_b : in0 (SrcB, lhs)
        0,                              // base_address_mask (unused, mask_chunk=false)
        0,                              // tile_index_a
        0,                              // tile_index_b
        params.TILE_SIZE_UNPACK_A,      // tile_size_a (in1 [32,32])
        params.TILE_SIZE_UNPACK_B,      // tile_size_b (in0 [M,32])
        KT_DIM,
        CT_DIM,
        false /* mask_chunk */);
}

#endif

#ifdef LLK_TRISC_MATH

// NOTE: llk_math_sdpa_custom_mm.h's _llk_math_sdpa_custom_mm_ declares
// `operandB_face_r_dim` but never reads it in either the fast (signal_granularity == ct_dim)
// or the general MOP/MVMUL path, so every instantiation trips -Werror=unused-parameter. This
// is a compile-level inconsistency with the canonical LLK header conventions (the param would
// be [[maybe_unused]] or dropped there), not a functional defect. We cannot edit the promoted
// header from a test, so we locally suppress the diagnostic across its include. The operand-B
// row count is still supplied via the runtime arg.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_sdpa_custom_mm.h"
#pragma GCC diagnostic pop
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // init: operandB_face_r_dim = in0 row count (M), ct_dim programs the MVMUL template.
    _llk_math_sdpa_custom_mm_init_<MM_TRANSPOSE>(params.in0_face_r_dim, CT_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // The sdpa math LLK zeroes DEST, then accumulates the whole kt walk into ct_dim tiles.
    _llk_math_sdpa_custom_mm_<SIGNAL_GRANULARITY>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM, false /* mask_chunk */);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // Mirror sdpa_custom_mm_block_init's PACK branch: standard pack hw_configure/init/dest
    // for [M,32] tiles, then the sdpa_custom_mm_block_init_pack_short() Z/W-stride RMWs that
    // pack the ct_dim result tiles densely (two half-tile faces per output).
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

    _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
        formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);

    // sdpa_custom_mm_block_init_pack_short(): Z-stride = FACE_C_DIM * 8 * 2,
    // W-stride = (TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2. Both are spelled here exactly
    // as the compute API spells them.
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2);

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t i = 0; i < CT_DIM; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // sdpa_custom_mm_block_uninit(): restore the default tile Z/W strides.
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * FACE_R_DIM * 2);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif

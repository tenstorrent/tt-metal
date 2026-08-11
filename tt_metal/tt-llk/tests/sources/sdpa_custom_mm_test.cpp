// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-fork experimental LLK sdpa_custom_mm (tt-metal#47554 / tt-blaze#1971), pending promotion.
// Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU signalling cadence
// (orthogonal to this numerical golden).
//
// The primitive under test is the demo-fork sdpa_custom_mm matmul (in0 [{1,2,4,8},32] -> SrcB, in1 [32,32] -> SrcA,
// rt_dim==1, ct_dim 1..16, kt_dim even 2..256, LoFi-only). We instantiate the mask_chunk=false path (per the
// comparison verdict) so no SrcB-mask / MOVB2D branch is taken and no base_address_mask buffer is required. MATH only
// POSTs semaphore::FPU_SFPU (non-blocking t6_semaphore_post) and never WAITS on SFPU, so the isolated UNPACK/MATH/PACK
// kernel does not deadlock; the un-drained FPU_SFPU increments are harmless.
//
// Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is pending
// Blackhole hardware/CI; this host is Wormhole.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved via the ADVANCE TEST shadow -I in test_config.py (demo-fork custom_mm root); repoint on promotion.
// llk_unpack_AB_sdpa_custom_mm.h transitively includes the demo-fork llk_unpack_AB_custom_mm.h (init + run helpers).
#include "llk_unpack_AB_sdpa_custom_mm.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // NOTE both operand-suffixed face counts are CROSSED below, like every other crossed argument in this call
    // (unpack_B_src/dst and TILE_SIZE_UNPACK_B go to the unpA slots, in0_face_r_dim goes to unpB_face_r_dim). The
    // harness defines num_faces_A as "active faces for matrix A", i.e. buffer_A == in0, so num_faces_A belongs in the
    // unpB slot and num_faces_B in the unpA slot. The silicon-validated matmul_custom_compressed_test.cpp:36-37
    // crosses them the same way, driven by NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4).
    // in0 goes to SrcB (partial tile [{1,2,4,8}, 32]), in1 goes to SrcA (full [32, 32]).
    // We keep buffer_A == in0 (SrcB) and buffer_B == in1 (SrcA); the primitive's address_a is SrcA and
    // address_b is SrcB, so the SrcA/SrcB arguments are crossed relative to the buffer names below.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src, // SrcA <- in1 (full tile)
        formats.unpack_A_src, // SrcB <- in0 (partial tile)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.in0_face_r_dim, // in0 partial-tile face row dim (rows in {1, 2, 4, 8})
        params.num_faces_B /* unpA_num_faces: in1, a full 4-face tile */,
        params.num_faces_A /* unpB_num_faces: in0, only the top two faces */,
        params.TILE_SIZE_UNPACK_B,  // SrcA tile size (in1)
        params.TILE_SIZE_UNPACK_A); // SrcB tile size (in0)

    // The sdpa unpack execute reuses the demo-fork custom_mm init (MOP config + counter setup).
    _llk_unpack_AB_custom_mm_init_<false /* transpose */>(params.in0_face_r_dim, formats.unpack_B_dst /* SrcA (in1) dst format */, params.CT_DIM);

    // sdpa variant takes an extra base_address_mask arg; mask_chunk defaults to false so it is never dereferenced.
    _llk_unpack_AB_sdpa_custom_mm_<false /* read_transposed */>(
        L1_ADDRESS(params.buffer_B[0]), // base_address_a -> SrcA (in1, full tile)
        L1_ADDRESS(params.buffer_A[0]), // base_address_b -> SrcB (in0, partial tile)
        0 /* base_address_mask (unused: mask_chunk == false) */,
        0 /* tile_index_a */,
        0 /* tile_index_b */,
        params.TILE_SIZE_UNPACK_B, // tile_size_a (in1)
        params.TILE_SIZE_UNPACK_A, // tile_size_b (in0)
        params.KT_DIM,
        params.CT_DIM,
        false /* mask_chunk */);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved via the ADVANCE TEST shadow -I in test_config.py (demo-fork custom_mm root); repoint on promotion.
//
// The demo-fork header trips two pre-existing -Werror warnings, both inert for numerics (see comparison verdict):
//   1. -Wunused-variable: a stray `uint32_t dst_face = dst_offset / 16;` in the mask_chunk branch of
//      _llk_math_sdpa_custom_mm_mask_dest_ (blaze deletes it).
//   2. -Wunused-parameter: `operandB_face_r_dim` of _llk_math_sdpa_custom_mm_ is never read (the mask/loops don't
//      consume it); the harness passes -Wunused-parameter explicitly.
// Both live inside template-free inline bodies reached by this TU, so suppress at file scope; an include-only wrap
// does not reach the definitions. Remove on promotion once the canonical header is warning-clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llk_math_common.h"
#include "llk_math_sdpa_custom_mm.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // sdpa_custom_mm is LoFi-only; init takes only transpose (no MathFidelity, split_acc, or dense_packing template).
    _llk_math_sdpa_custom_mm_init_<false /* transpose */>(params.in0_face_r_dim, params.CT_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // mask_chunk == false: skip the SrcB-mask / MOVB2D path (ZEROACC-only DEST clear), per the comparison verdict.
    _llk_math_sdpa_custom_mm_(params.in0_face_r_dim, 0 /* dst_index */, params.KT_DIM, params.CT_DIM, false /* mask_chunk */);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Partial-tile pack. sdpa_custom_mm's addrmod hardcodes face_r_dim == 8 and steps DEST by 8 within a tile and 16
    // between tiles, so each output tile is exactly ONE 16x16 DEST face carrying an 8x32 logical tile (rows 0-7 are
    // logical columns 0-15, rows 8-15 are columns 16-31). The packer therefore runs single-face, and the DEST
    // tile-to-tile stride comes down from 64 rows to 16, i.e. Wstride / 4. TILE_SIZE_PACK stays the full 32x32 tile so
    // the L1 stride matches what the host reads back; the host keeps the leading face of each tile.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, FACE_R_DIM, FACE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, FACE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 4) * FACE_C_DIM * FACE_R_DIM * 2);
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif

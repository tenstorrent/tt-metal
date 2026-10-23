// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers experimental LLK compressed_custom_mm (tt-metal#47554 / tt-blaze#1971), promoted into
// tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The includes below resolve through the canonical -I; the
// demo-fork shadow tree this test was first written against no longer exists.

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
// Resolved from the promoted experimental/ copy (landed on main via #53295); the demo-fork shadow tree this test was
// originally written against is gone. No -Wunused shim needed on this thread either: the promoted header reads every
// parameter it takes, so it builds clean under -Werror -Wunused-parameter.
#include "experimental/llk_unpack_AB_compressed_custom_mm.h"
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
    // in0 goes to SrcB (partial tile [{1,2,4,8}, 32], bf16), in1 goes to SrcA (full [32, 32], BFP-compressed).
    // buffer_A == in0 (SrcB), buffer_B == in1 (SrcA); the primitive's address_a is SrcA and address_b is SrcB, so the
    // SrcA/SrcB arguments are crossed relative to the buffer names below (same crossing as matmul_custom_compressed_test.cpp).
    // buffer_C carries the per-tile compression metadata (packed 3-bit format codes) read by the primitive.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src, // SrcA <- in1 (full tile, BFP-compressed)
        formats.unpack_A_src, // SrcB <- in0 (partial tile, bf16)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.in0_face_r_dim, // in0 partial-tile face row dim (rows in {1, 2, 4, 8})
        params.num_faces_B /* unpA_num_faces: in1, a full 4-face tile */,
        params.num_faces_A /* unpB_num_faces: in0, only the top two faces */,
        params.TILE_SIZE_UNPACK_B,  // SrcA tile size (in1)
        params.TILE_SIZE_UNPACK_A); // SrcB tile size (in0)

    // compressed_custom_mm unpack init takes only unpB_face_r_dim (no CT_DIM, unlike custom_mm).
    _llk_unpack_AB_compressed_custom_mm_init_<false /* transpose */>(params.in0_face_r_dim);

    _llk_unpack_AB_compressed_custom_mm_<true /* clear_src */>(
        L1_ADDRESS(params.buffer_B[0]), // base_address_a -> SrcA (in1, BFP-compressed full tile)
        L1_ADDRESS(params.buffer_A[0]), // base_address_b -> SrcB (in0, partial tile)
        params.buffer_C[0],             // base_address_meta -> per-tile compression metadata. NOT L1_ADDRESS(): the primitive
                                        // dereferences this on the RISC-V core (meta_ptr[i]), so it needs the raw byte address,
                                        // not the /16 Tensix unpacker encoding.
        params.KT_DIM,
        params.CT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry), from the promoted
// experimental/ copy. No -Wunused shim needed on this thread: the promoted math header is warning-clean on the path
// we instantiate.
#include "experimental/llk_math_compressed_custom_mm.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // compressed_custom_mm is LoFi-only; init takes no MathFidelity template.
    // dense_packing == true brings the DEST tile-to-tile stride down from 64 rows to 32, matching the partial-tile
    // packer below and the silicon-validated matmul_custom_compressed_test.cpp:62.
    _llk_math_compressed_custom_mm_init_<false /* transpose */, false /* split_acc */, true /* dense_packing */>(params.in0_face_r_dim);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // finalize == false because split_acc == false (see compressed_custom_mm.h contract: finalize must be false if
    // split_acc is false). The math primitive reads the same metadata buffer as the unpacker.
    _llk_math_compressed_custom_mm_<false /* finalize */>(
        params.buffer_C[0], // base_address_meta -- raw byte address, see the unpack-side note above
        params.in0_face_r_dim,
        0 /* dst_index */,
        params.KT_DIM,
        params.CT_DIM);

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
    // Partial-tile pack, copied from the silicon-validated matmul_custom_compressed_test.cpp:85-91. The output tile
    // is [in0_face_r_dim, 32]: two faces of in0_face_r_dim x 16, each face still occupying a full 16-row DEST slot,
    // and with dense_packing the DEST tile-to-tile stride is 32 rows instead of 64 -- hence Wstride / 2. Without this
    // the packer emits full 32x32 tiles, so only the top in0_face_r_dim rows of each face carry the result and the
    // rest is whatever DEST held.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true /* partial_face */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.in0_face_r_dim, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif

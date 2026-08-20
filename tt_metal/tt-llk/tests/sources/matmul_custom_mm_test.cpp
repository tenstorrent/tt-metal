// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Plain (uncompressed) custom_mm matmul -- experimental, Blackhole only.
//
// Why this exists. The plain custom_mm family was promoted to experimental/ by tt-metal
// #52727 and, unlike its compressed sibling, had **no test at all**: all four entry points
// (_llk_unpack_AB_custom_mm_init_, _llk_unpack_AB_custom_mm_, _llk_math_custom_mm_init_,
// _llk_math_custom_mm_) were uncalled anywhere under tests/sources. The asymmetry is easy
// to miss -- matmul_custom_compressed_test.cpp covers the *compressed* variant, so the
// harder path was exercised and the simpler one was not, and matmul_custom_test.cpp drives
// llk_math_matmul_custom_no_mop.h, an unrelated family. Tracked as A1.
//
// Shape of the op. This is not a general matmul. Operand a is a full 32x32 4-face tile;
// operand b is a narrow [{1,2,4,8}, 32] tile with only its top two faces, unpacked one
// face per instruction (see the SETADCXX pair in _llk_unpack_AB_custom_mm_init_). So the
// computation is (M x K) @ (K x N) with M in {1,2,4,8} -- a narrow activation row times a
// full weight block -- accumulating over kt_dim K-tiles into ct_dim output tiles.
//
// The whole kt loop runs from ONE call on each thread, which is the point of the family:
// the unpack side issues a single TT_MOP for up to 256 K-tiles and the math side a single
// ckernel_template. That also fixes the parity constraint the doc tables state --
// _llk_unpack_AB_custom_mm_run_ issues TT_MOP(0, (kt_dim / 2) - 1, 0), so **kt_dim must be
// even and at least 2**.
//
// Operand order is swapped on purpose, matching the compute API's "Intentionally swap in0
// and in1 as operation specific hw_configures are deprecated" and the compressed driver:
// the full tiles live in buffer_B and are passed as base_address_a, the narrow rows live in
// buffer_A and are passed as base_address_b.
//
// dense_packing is used here, as in matmul_custom_compressed_test.cpp: the math init packs
// the ct output tiles 32 rows apart rather than 64, so the pack side programs the matching
// W-stride and restores it afterwards. That mirrors what a real caller does via
// custom_mm_block_init / _block_uninit.
//
// Not covered yet, deliberately: transpose, split_acc /
// finalize (both ARE forwarded on this family, unlike the compressed one), and the top of
// the documented kt_dim range. This file establishes the family under test; widening it is
// cheaper than starting it.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // Swapped, as above: unpacker A is configured from the B buffer (full tiles) and
    // unpacker B from the A buffer (narrow rows).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,
        formats.unpack_A_src,
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        params.in1_face_r_dim,
        params.in0_face_r_dim,
        params.num_faces_B,
        params.num_faces_A,
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);

    // unpA_dst_format only selects a profiling heuristic (post1 for Bfp4_b), so the
    // unpacker-A destination format is what to hand it.
    _llk_unpack_AB_custom_mm_init_<false /* transpose */>(params.in0_face_r_dim, formats.unpack_B_dst, CT_DIM);

    _llk_unpack_AB_custom_mm_<false /* read_transposed */, true /* clear_src */>(
        L1_ADDRESS(params.buffer_B[0]),
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index_a */,
        0 /* tile_index_b */,
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A,
        KT_DIM,
        CT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_custom_mm.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_custom_mm_init_<false /* transpose */, false /* split_acc */, true /* dense_packing */>(params.in0_face_r_dim, CT_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    _llk_math_custom_mm_<false /* finalize */>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

    _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
        formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);

    // dense_packing: the math init laid the ct output tiles 32 rows apart, so the packer
    // needs the matching stride. Restored after the packs, exactly as *_block_uninit does.
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t i = 0; i < CT_DIM; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif

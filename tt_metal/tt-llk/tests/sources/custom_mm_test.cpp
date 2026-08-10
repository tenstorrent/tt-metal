// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-tree experimental LLK custom_mm (tt-metal#47554 / tt-blaze#1971), pending promotion into
// tt_llk_blackhole/llk_lib/experimental/. Include path below must be repointed to the canonical header on promotion.
// Primitive verified byte-identical to tt-blaze main as of this writing.

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
// On promotion, repoint the -I in test_config.py so this resolves to the canonical header and this line is unchanged.
#include "llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // in0 goes to SrcB (partial tile [{1,2,4,8}, 32]), in1 goes to SrcA (full [32, 32]).
    // We keep buffer_A == in0 (SrcB) and buffer_B == in1 (SrcA); the primitive's address_a is SrcA and
    // address_b is SrcB, so the SrcA/SrcB arguments are crossed relative to the buffer names below.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,  // SrcA <- in1 (full tile)
        formats.unpack_A_src,  // SrcB <- in0 (partial tile)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.IN0_FACE_R_DIM,  // in0 partial-tile face row dim (rows in {1, 2, 4, 8})
        params.num_faces_A /* unpA_num_faces (in1, full tile) */,
        params.num_faces_B /* in0 active faces */,
        params.TILE_SIZE_UNPACK_B,   // SrcA tile size (in1)
        params.TILE_SIZE_UNPACK_A);  // SrcB tile size (in0)

    _llk_unpack_AB_custom_mm_init_<false /* transpose */>(
        params.IN0_FACE_R_DIM, formats.unpack_B_dst /* SrcA (in1) dst format */, params.CT_DIM);

    _llk_unpack_AB_custom_mm_<false /* read_transposed */, true /* clear_src */>(
        L1_ADDRESS(params.buffer_B[0]),  // base_address_a -> SrcA (in1, full tile)
        L1_ADDRESS(params.buffer_A[0]),  // base_address_b -> SrcB (in0, partial tile)
        0 /* tile_index_a */,
        0 /* tile_index_b */,
        params.TILE_SIZE_UNPACK_B,  // tile_size_a (in1)
        params.TILE_SIZE_UNPACK_A,  // tile_size_b (in0)
        params.KT_DIM,
        params.CT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
#include "llk_math_common.h"
#include "llk_math_custom_mm.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // custom_mm is LoFi-only; init takes no MathFidelity template.
    _llk_math_custom_mm_init_<false /* transpose */, false /* split_acc */, false /* dense_packing */>(
        params.IN0_FACE_R_DIM, params.CT_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // finalize == false because split_acc == false (see custom_mm.h contract: finalize must be false if split_acc is false).
    _llk_math_custom_mm_<false /* finalize */>(params.IN0_FACE_R_DIM, 0 /* dst_index */, params.KT_DIM, params.CT_DIM);

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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-tree experimental LLK compressed_custom_mm (tt-metal#47554 / tt-blaze#1971), pending
// promotion into tt_llk_blackhole/llk_lib/experimental/. Include path (shared with custom_mm) must be repointed on
// promotion. Primitives verified byte-identical to tt-blaze main as of this writing.

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
//
// The unpromoted demo-tree header carries constexpr-computed locals (get_replay_insn_for_combo intermediates) and a
// clear_src template branch that the harness's -Werror -Wunused-variable / -Wunused-parameter flags on some
// instantiations. Suppress locally around the include (do NOT edit the byte-identical shadow header); remove on
// promotion once the canonical header is warning-clean.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llk_unpack_AB_compressed_custom_mm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // in0 goes to SrcB (partial tile [{1,2,4,8}, 32], bf16), in1 goes to SrcA (full [32, 32], BFP-compressed).
    // buffer_A == in0 (SrcB), buffer_B == in1 (SrcA); the primitive's address_a is SrcA and address_b is SrcB, so the
    // SrcA/SrcB arguments are crossed relative to the buffer names below (matches custom_mm_test.cpp).
    // buffer_C carries the per-tile compression metadata (packed 3-bit format codes) read by the primitive.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,  // SrcA <- in1 (full tile, BFP-compressed)
        formats.unpack_A_src,  // SrcB <- in0 (partial tile, bf16)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.IN0_FACE_R_DIM,  // in0 partial-tile face row dim (rows in {1, 2, 4, 8})
        params.num_faces_A /* unpA_num_faces (in1, full tile) */,
        params.num_faces_B /* in0 active faces */,
        params.TILE_SIZE_UNPACK_B,   // SrcA tile size (in1)
        params.TILE_SIZE_UNPACK_A);  // SrcB tile size (in0)

    // compressed_custom_mm unpack init takes only unpB_face_r_dim (no CT_DIM, unlike custom_mm).
    _llk_unpack_AB_compressed_custom_mm_init_<false /* transpose */>(params.IN0_FACE_R_DIM);

    _llk_unpack_AB_compressed_custom_mm_<true /* clear_src */>(
        L1_ADDRESS(params.buffer_B[0]),  // base_address_a -> SrcA (in1, BFP-compressed full tile)
        L1_ADDRESS(params.buffer_A[0]),  // base_address_b -> SrcB (in0, partial tile)
        L1_ADDRESS(params.buffer_C[0]),  // base_address_meta -> per-tile compression metadata
        params.KT_DIM,
        params.CT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// See the unpack-side note above: the unpromoted demo-tree header has constexpr/branch-dependent locals that trip the
// harness's -Werror on some instantiations. Suppress at file scope (the offending vars live inside template bodies, so
// an include-only wrap does not reach the instantiation point). Remove on promotion once the canonical header is clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "llk_math_common.h"
#include "llk_math_compressed_custom_mm.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // compressed_custom_mm is LoFi-only; init takes no MathFidelity template.
    _llk_math_compressed_custom_mm_init_<false /* transpose */, false /* split_acc */, false /* dense_packing */>(
        params.IN0_FACE_R_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // finalize == false because split_acc == false (see compressed_custom_mm.h contract: finalize must be false if
    // split_acc is false). The math primitive reads the same metadata buffer as the unpacker.
    _llk_math_compressed_custom_mm_<false /* finalize */>(
        L1_ADDRESS(params.buffer_C[0]),  // base_address_meta
        params.IN0_FACE_R_DIM,
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

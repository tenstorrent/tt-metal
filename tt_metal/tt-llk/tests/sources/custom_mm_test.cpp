// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Plain (non-compressed) custom_mm matmul, driving the experimental Blackhole-only
// LLK pipeline promoted by tt-metal #52727:
//
//   tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_custom_mm.h
//   tt_llk_blackhole/llk_lib/experimental/llk_math_custom_mm.h
//   hw/inc/api/compute/experimental/custom_mm.h   (custom_mm_block / _init / _unpack / _math)
//
// llk_math_custom_mm.h has no tt-llk coverage today; this is the first driver of the
// PLAIN path (matmul_custom_compressed_test.cpp already exercises the *compressed* sibling
// llk_{unpack_AB,math}_compressed_custom_mm.h, and this file mirrors its structure with
// the compression/meta stripped out).
//
// ---------------------------------------------------------------------------------------
// GOLDEN DERIVATION (so a BH reviewer can check it against the header math)
// ---------------------------------------------------------------------------------------
// custom_mm computes a standard tiled matmul C = A * B, C[M,N] = A[M,K] * B[K,N], with the
// operand-to-source-register mapping the compute API bakes in (custom_mm.h line 57:
// llk_unpack_hw_configure(in1_cb_id, in0_cb_id), i.e. in1->SrcA, in0->SrcB):
//
//   in0 (A, the [M,K] operand)  -> SrcB, tile shape [{1,2,4,8}, 32]  (only the top two
//                                  faces, each face_r_dim rows tall; face_r_dim = M)
//   in1 (B, the [K,N] operand)  -> SrcA, full [32,32] tiles
//   rt_dim = 1, ct_dim in [1,16], kt_dim in [1,256], LoFi only.
//
// So the header restricts M to {1,2,4,8} rows: only those DEST rows are defined. The math
// LLK (_llk_math_custom_mm_) runs one MVMUL walk per k-tile, accumulating A*B into DEST;
// with split_acc=false / finalize=false there is NO finalization pass (finalize must be
// false when split_acc is false -- custom_mm.h arg table, line 138), so DEST holds the
// plain accumulated product. The result is packed as ct_dim output tiles; inside each tile
// the two 16-col faces (M x 16, row-major) sit contiguously, then pad out to a full tile.
//
// The Python golden is exactly torch A[M,K] @ B[K,N] (MatmulGolden, LoFi), and it asserts
// ONLY the M defined rows of each output tile against that product (the rest of each 32-row
// DEST tile is undefined and is not compared) -- see test_custom_mm.py. This mirrors the
// header math: the LLK only writes the top face_r_dim=M rows of each output face.
//
// Buffer layout the driver assumes (matches helpers/compressed_utils.run_compressed, which
// is the established layout for this exact tile geometry):
//   buffer_A (SrcB, in0/A): kt*2 faces of [M,16], face-column order along K, contiguous.
//   buffer_B (SrcA, in1/B): kt*ct full [32,32] tiles, k-major / c-minor (the order the
//                           SrcA CFGSHIFTMASK walk reads them with read_transposed=false,
//                           block_increment = inner_increment = tile_size).
//
// The single unpack call issues one MOP covering the whole kt walk; the single math call
// loops kt internally -- both are invoked ONCE (not per k-tile), exactly as the LLK's own
// run wrappers do (_llk_unpack_AB_custom_mm_run_ issues TT_MOP once for kt/2-1; the finalize
// path is disabled here).

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// K splits over CUSTOM_MM_NUM_CALLS back-to-back calls that accumulate into one DEST; 1 is the single-call kernel.
// More than one call is for test_custom_mm_multi_call, which exercises races at the call boundary, where the
// unpacker enters call c + 1 while math still works on call c. The result must not depend on that timing.
constexpr std::uint32_t num_calls   = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call = KT_DIM / num_calls;
static_assert(kt_per_call * num_calls == KT_DIM, "K tiles must split evenly across the calls");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // in1 (B, full [32,32] tiles) is SrcA; in0 (A, [M,32] tiles) is SrcB.
    // hw_configure takes the SrcA operand first, so pass the B-matrix (buffer_B) params
    // in the A slot and the A-matrix (buffer_A) params in the B slot -- exactly the
    // in0/in1 swap custom_mm.h line 57 performs.
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

    if constexpr (num_calls > 1)
    {
        // Race cases only. Leave a non-zero SrcA clear value on unpacker 0: a both-bank SrcB clear only corrupts
        // the SrcA writes it overlaps while unpacker 0's last SrcA clear value is non-zero, and that value outlives
        // kernels. Without this the race would pass or fail depending on what last ran on the core.
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
    }

    // unpB_face_r_dim = in0 (SrcB) face rows in {1,2,4,8}; unpA_dst_format tunes the
    // instruction sequence (post1 only for Bfp4_b). transpose=false.
    _llk_unpack_AB_custom_mm_init_<false /* transpose */, true /* clear_src */>(params.in0_face_r_dim, formats.unpack_B_dst, CT_DIM);

    // SrcA=buffer_B (B matrix, full tiles), SrcB=buffer_A (A matrix). Call c takes K tiles
    // [c * kt_per_call, (c + 1) * kt_per_call): B is k-major, so its tiles start at c * kt_per_call * ct,
    // and an in0 tile is 4 * in0_face_r_dim 16-byte words (TILE_SIZE_UNPACK_A is a full tile here).
    // Within a call the SrcA walk covers its kt*ct grid via CFGSHIFTMASK, the SrcB walk covers kt via counters.
    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        _llk_unpack_AB_custom_mm_<false /* read_transposed */>(
            L1_ADDRESS(params.buffer_B[0]),
            L1_ADDRESS(params.buffer_A[0]) + call * kt_per_call * 4 * params.in0_face_r_dim,
            call * kt_per_call * CT_DIM /* tile_index_a */,
            0 /* tile_index_b */,
            params.TILE_SIZE_UNPACK_B,
            params.TILE_SIZE_UNPACK_A,
            kt_per_call,
            CT_DIM);
    }
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

    if constexpr (CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG)
    {
        // Leave the Src zero flag at keep, as a preceding copy_tile_init does. The init must restore the
        // operand-driven value, or MVMUL keeps the denormal SrcB operands it should flush.
        math::_configure_preserve_zero_flag_state_();
    }

    // split_acc=false, dense_packing=true, transpose=false. operandB_face_r_dim = in0 M.
    // dense_packing MUST be true: the pack thread reads the ct output tiles with the
    // dense-packing W-stride (consecutive tiles 32 DEST rows apart), so the math must lay
    // them out 32 rows apart too (ADDR_MOD_2 DEST incr = 32, not 64). With dense_packing
    // false the math strides output tiles 64 rows apart, so pack reads tile i>0 from the
    // wrong DEST offset -- tile 0 is correct but every later tile is corrupt. Matches the
    // proven compressed sibling (matmul_custom_compressed_test.cpp), which passes
    // dense_packing=true for the identical pack setup.
    _llk_math_custom_mm_init_<false /* transpose */, false /* split_acc */, true /* dense_packing */>(params.in0_face_r_dim, CT_DIM);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // finalize MUST be false when split_acc is false (custom_mm.h arg table): with no split
    // accumulation there are no partials to merge, and the DEST already holds A*B.
    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        if constexpr (num_calls > 1)
        {
            // Race cases only. Hold math back before each call so the unpacker runs ahead and reaches call c + 1
            // while math still holds call c's last SrcA and SrcB banks. Anything call c + 1 issues before its
            // unpacks, such as a both-bank SrcB clear, then starts on the same bank release as its first SrcA
            // writes. Without the delay math keeps up with the unpacker and the boundary is never contended.
            ckernel::wait(2000);
        }
        _llk_math_custom_mm_<false /* finalize */>(params.in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM);
    }

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

    // Same pack setup the compressed driver uses: the in0 face geometry (M rows, 2 faces)
    // is what lands in DEST, so pack with that geometry and the dense-packing W-stride
    // (tiles 32 rows apart) that custom_mm_block_init installs for the [M,32] output tiles.
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

    _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
        formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * (is_fp32_dest_acc_en ? 4 : 2));

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t i = 0; i < CT_DIM; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * (is_fp32_dest_acc_en ? 4 : 2));
}

#endif

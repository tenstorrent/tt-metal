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
//   in1 (B, the [K,N] operand)  -> SrcA, full [32,32] tiles, or [32,16] ones (num_faces_B = 2)
//                                  giving [M,16] output tiles (num_faces = 1)
//   rt_dim = 1, ct_dim in [1,34] (bounded by DEST capacity), kt_dim in [1,256], LoFi only.
//
// So the header restricts M to {1,2,4,8} rows: only those DEST rows are defined. The math
// LLK (_llk_math_custom_mm_) runs one MVMUL walk per k-tile, accumulating A*B into DEST.
// The SplitAcc layouts accumulate the second k-face apart and the last call finalizes, merging
// it back; the others need no finalization pass, so DEST holds the plain accumulated product
// either way. The result is packed as ct_dim output tiles; inside each tile its 16-col faces
// (M x 16, row-major: two, or one for [M,16] tiles) sit contiguously, then pad out to a full tile.
//
// The Python golden is exactly torch A[M,K] @ B[K,N] (MatmulGolden, LoFi), and it asserts
// ONLY the M defined rows of each output tile against that product (the rest of each 32-row
// DEST tile is undefined and is not compared) -- see test_custom_mm.py. This mirrors the
// header math: the LLK only writes the top face_r_dim=M rows of each output face.
//
// Buffer layout the driver assumes (matches helpers/compressed_utils.run_compressed, which
// is the established layout for this exact tile geometry):
//   buffer_A (SrcB, in0/A): kt*2 faces of [M,16], face-column order along K, contiguous.
//   buffer_B (SrcA, in1/B): kt*ct [32,32] or [32,16] tiles, k-major / c-minor (the order the
//                           SrcA CFGSHIFTMASK walk reads them with read_transposed=false,
//                           block_increment = inner_increment = tile_size). With
//                           CUSTOM_MM_TRANSPOSE each tile holds B^T instead ([32,32] or [16,32]).
//
// The single unpack call issues one MOP covering the whole kt walk; the single math call
// loops kt internally -- both are invoked ONCE (not per k-tile), exactly as the LLK's own
// run wrappers do (_llk_unpack_AB_custom_mm_run_ issues one TT_MOP iteration per k-tile; the
// finalize path is disabled here).

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "experimental/llk_custom_mm_common.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

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

// CUSTOM_MM_LAYOUT picks where the output tiles sit in DEST, the same on the math and pack threads. The SplitAcc
// layouts finalize on the last call.
static_assert(CUSTOM_MM_LAYOUT <= static_cast<std::uint8_t>(ckernel::CustomMmDestLayout::Compact), "CUSTOM_MM_LAYOUT is not a CustomMmDestLayout");
constexpr auto dest_layout = static_cast<ckernel::CustomMmDestLayout>(CUSTOM_MM_LAYOUT);
constexpr bool split_acc   = ckernel::custom_mm_split_acc(dest_layout);

// CUSTOM_MM_TRANSPOSE: each in1 tile holds B^T ([N, 32], or [16, 32] for a narrow tile), and the unpacker transposes
// each face back. It only selects the init's face transpose and face order; the weight tiles are still walked
// k-major, so read_transposed stays false.
constexpr bool transpose = CUSTOM_MM_TRANSPOSE;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;
    const std::uint32_t in0_face_r_dim     = params.in0_face_r_dim;
    const std::uint32_t in1_face_r_dim     = params.in1_face_r_dim;
    const auto& buffer_A                   = params.buffer_A;
    const auto& buffer_B                   = params.buffer_B;
#endif

    const std::uint32_t in1_tile_size = TILE_SIZE_UNPACK_B / TILE_NUM_FACES * num_faces_B;

    {
        START_PERF_MEASURE("INIT")
        // hw_configure takes SrcA first, so B's parameters occupy its A slot.
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            in1_face_r_dim,
            in0_face_r_dim,
            num_faces_B,
            num_faces_A,
            in1_tile_size,
            TILE_SIZE_UNPACK_A);

        if constexpr (num_calls > 1)
        {
            // Make the persistent SrcA clear value deterministic for the race cases.
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
        }

        _llk_unpack_AB_custom_mm_init_<transpose, true /* clear_src */>(in0_face_r_dim, formats.unpack_B_dst, CT_DIM, num_faces_B);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Match each k-tile's one SrcA+SrcB post and CT_DIM-1 SrcA-only reuses.
            for (std::uint32_t k = 0; k < LOOP_FACTOR * num_calls * kt_per_call; ++k)
            {
                _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(1 /* iterations */);
                if constexpr (CT_DIM > 1)
                {
                    _perf_unpack_stream_set_valid<true /* set_a */, false /* set_b */, PERF_RUN_TYPE>(CT_DIM - 1 /* iterations */);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t call = 0; call < num_calls; call++)
                {
                    _llk_unpack_AB_custom_mm_<false /* read_transposed */>(
                        L1_ADDRESS(buffer_B[0]),
                        L1_ADDRESS(buffer_A[0]) + call * kt_per_call * 4 * in0_face_r_dim,
                        call * kt_per_call * CT_DIM /* tile_index_a */,
                        0 /* tile_index_b */,
                        in1_tile_size,
                        TILE_SIZE_UNPACK_A,
                        kt_per_call,
                        CT_DIM);
                }
            }
        }
        PROFILER_SYNC();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t in0_face_r_dim = params.in0_face_r_dim;
    const std::uint32_t num_faces_B    = params.num_faces_B;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        if constexpr (CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG)
        {
            // Recreate a preceding copy_init state; custom_mm init must restore zeroing.
            math::_configure_preserve_zero_flag_state_();
        }

        _llk_math_custom_mm_init_<transpose, dest_layout>(in0_face_r_dim, CT_DIM, num_faces_B);

        LLK_ASSERT((custom_mm_fits_dest<dest_sync, dest_layout>(0 /* dst_index */, CT_DIM)), "custom_mm: CT_DIM output tiles overrun the DEST sync region");
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // Math clears SrcA on the first CT_DIM-1 tiles and SrcA+SrcB on the last.
            for (std::uint32_t k = 0; k < LOOP_FACTOR * num_calls * kt_per_call; ++k)
            {
                if constexpr (CT_DIM > 1)
                {
                    _perf_math_stream_clear_valid<true /* clear_a */, false /* clear_b */, PERF_RUN_TYPE>(CT_DIM - 1 /* iterations */);
                }
                _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(1 /* iterations */);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                }
                for (std::uint32_t call = 0; call < num_calls; call++)
                {
#ifndef LLK_PROFILER
                    if constexpr (num_calls > 1)
                    {
                        // Race cases only: let unpack enter the next call while math holds the prior banks.
                        ckernel::wait(2000 /* cycles */);
                    }
#endif
                    // Only SplitAcc has partials to merge, on the iteration's last call.
                    if (split_acc && call == num_calls - 1)
                    {
                        _llk_math_custom_mm_<dest_layout>(in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM, num_faces_B);
                    }
                    else
                    {
                        _llk_math_custom_mm_<dest_layout, false /* finalize */>(in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM, num_faces_B);
                    }
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_custom_mm.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t in0_face_r_dim = params.in0_face_r_dim;
    const std::uint32_t num_faces      = params.num_faces;
    const auto& buffer_Res             = params.buffer_Res;
#endif

    const std::uint32_t out_tile_c_dim = num_faces * FACE_C_DIM;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_face_r_dim, out_tile_c_dim, num_faces, true);

        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
            formats.pack_src, in0_face_r_dim, out_tile_c_dim, num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        _llk_pack_custom_mm_init_<dest_layout>(formats.pack_src);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(buffer_Res[i]));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_pack_custom_mm_uninit_<dest_layout>(formats.pack_src);
}

#endif

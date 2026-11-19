// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers experimental compressed_custom_mm through its canonical headers.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "compressed_mm_perf_helpers.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

inline std::uint32_t compressed_meta_format_code(const std::uint32_t format)
{
    switch (static_cast<DataFormat>(format))
    {
        case DataFormat::Bfp8_b:
            return COMPRESSED_META_BFP8_CODE;
        case DataFormat::Bfp4_b:
            return COMPRESSED_META_BFP4_CODE;
        case DataFormat::Bfp2_b:
            return COMPRESSED_META_BFP2_CODE;
        default:
            return 0;
    }
}

// Perf runs seed metadata locally before the profiler rendezvous; functional runs use host metadata.
inline void initialize_perf_compressed_meta(const std::uint32_t meta_base, const std::uint32_t format, const std::uint32_t kt_dim, const std::uint32_t ct_dim)
{
#ifdef LLK_PROFILER
    volatile std::uint32_t* meta_ptr = reinterpret_cast<volatile std::uint32_t*>(meta_base);
    const std::uint32_t fmt          = compressed_meta_format_code(format);
    const std::uint32_t tile_count   = kt_dim * ct_dim;
    std::uint32_t tile               = 0;
    std::uint32_t prev_fmt           = 0;

    for (std::uint32_t word = 0; tile < tile_count; ++word)
    {
        std::uint32_t encoded = prev_fmt;
        for (std::uint32_t slot = 0; slot < COMPRESSED_META_SLOTS_PER_WORD && tile < tile_count; ++slot, ++tile)
        {
            const std::uint32_t use_b = (tile % ct_dim) == 0 ? 1 : 0;
            encoded |= use_b << (COMPRESSED_META_BITS_PER_SLOT * slot + COMPRESSED_META_USE_B_OFFSET);
            encoded |= fmt << (COMPRESSED_META_BITS_PER_SLOT * slot + COMPRESSED_META_FORMAT_OFFSET);
            prev_fmt = fmt;
        }
        meta_ptr[word] = encoded;
    }
#else
    (void)meta_base;
    (void)format;
    (void)kt_dim;
    (void)ct_dim;
#endif
}

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t KT_DIM             = params.KT_DIM;
    const std::uint32_t CT_DIM             = params.CT_DIM;
    const std::uint32_t in0_face_r_dim     = params.in0_face_r_dim;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const auto& buffer_A                   = params.buffer_A;
    const auto& buffer_B                   = params.buffer_B;
    const auto& buffer_C                   = params.buffer_C;
#endif

    initialize_perf_compressed_meta(buffer_C[0], formats.unpack_B_src, KT_DIM, CT_DIM);

    {
        START_PERF_MEASURE("INIT")
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
            in0_face_r_dim, // in0 partial-tile face row dim (rows in {1, 2, 4, 8})
            num_faces_B /* unpA_num_faces: in1, a full 4-face tile */,
            num_faces_A /* unpB_num_faces: in0, only the top two faces */,
            TILE_SIZE_UNPACK_B,  // SrcA tile size (in1)
            TILE_SIZE_UNPACK_A); // SrcB tile size (in0)

        // compressed_custom_mm unpack init takes only unpB_face_r_dim (no CT_DIM, unlike custom_mm).
        _llk_unpack_AB_compressed_custom_mm_init_<false /* transpose */, true /* clear_src */>(in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            compressed_mm_isolate_dvalids<true>(LOOP_FACTOR, 1 /* num_calls */, KT_DIM, CT_DIM, buffer_C[0], 0 /* meta_bytes_per_call */);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_AB_compressed_custom_mm_(
                    L1_ADDRESS(buffer_B[0]), // base_address_a -> SrcA (in1, BFP-compressed full tile)
                    L1_ADDRESS(buffer_A[0]), // base_address_b -> SrcB (in0, partial tile)
                    buffer_C[0],             // raw byte address: metadata is read by the RISC-V core
                    KT_DIM,
                    CT_DIM);
            }
        }
        PROFILER_SYNC();
    }
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const auto& buffer_C            = params.buffer_C;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        // compressed_custom_mm is LoFi-only; init takes no MathFidelity template.
        // The Dense layout brings the DEST tile-to-tile stride down from 64 rows to 32, matching the partial-tile packer
        // below and the silicon-validated matmul_custom_compressed_test.cpp.
        _llk_math_compressed_custom_mm_init_<false /* transpose */, CustomMmDestLayout::Dense>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            compressed_mm_isolate_dvalids<false>(LOOP_FACTOR, 1 /* num_calls */, KT_DIM, CT_DIM, buffer_C[0], 0 /* meta_bytes_per_call */);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }

                // The math primitive reads the same metadata buffer as the unpacker.
                _llk_math_compressed_custom_mm_<CustomMmDestLayout::Dense>(
                    buffer_C[0], // base_address_meta -- raw byte address, see the unpack-side note above
                    0 /* dst_index */,
                    KT_DIM,
                    CT_DIM);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT       = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t in0_face_r_dim = params.in0_face_r_dim;
    const std::uint32_t num_faces      = params.num_faces;
    const auto& buffer_Res             = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        // Partial-tile pack, copied from the silicon-validated matmul_custom_compressed_test.cpp:85-91. The output tile
        // is [in0_face_r_dim, 32]: two faces of in0_face_r_dim x 16, each face still occupying a full 16-row DEST slot,
        // and in the Dense layout the DEST tile-to-tile stride is 32 rows instead of 64 -- hence Wstride / 2. Without this
        // the packer emits full 32x32 tiles, so only the top in0_face_r_dim rows of each face carry the result and the
        // rest is whatever DEST held.
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_face_r_dim, TILE_C_DIM, num_faces, true /* partial_face */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, in0_face_r_dim, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_custom_mm_init_<CustomMmDestLayout::Dense>(formats.pack_src);
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
                for (std::uint32_t i = 0; i < TILE_CNT; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(buffer_Res[i]));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_pack_custom_mm_uninit_<CustomMmDestLayout::Dense>(formats.pack_src);
}

#endif

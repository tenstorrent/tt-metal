// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "compressed_mm_perf_helpers.h"
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

// Split K across back-to-back calls that accumulate into one DEST.
constexpr std::uint32_t num_calls           = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call         = KT_DIM / num_calls;
constexpr std::uint32_t bfp8_tile_words     = 68;
constexpr std::uint32_t b_words_per_call    = kt_per_call * CT_DIM * bfp8_tile_words;
constexpr std::uint32_t meta_bytes_per_call =
    ((kt_per_call * CT_DIM + COMPRESSED_META_SLOTS_PER_WORD - 1) / COMPRESSED_META_SLOTS_PER_WORD) * sizeof(std::uint32_t);
static_assert(kt_per_call * num_calls == KT_DIM && kt_per_call % 2 == 0, "each call needs an even share of the K tiles");

// Perf runs copy compile-time metadata locally before the profiler rendezvous.
inline void initialize_perf_compressed_meta(const std::uint32_t meta_base)
{
#ifdef LLK_PROFILER
    static_assert(
        PERF_COMPRESSED_META_WORD_COUNT == num_calls * meta_bytes_per_call / sizeof(std::uint32_t),
        "perf metadata size does not match the compressed call layout");
    volatile std::uint32_t* meta_ptr = reinterpret_cast<volatile std::uint32_t*>(meta_base);
    for (std::uint32_t word = 0; word < PERF_COMPRESSED_META_WORD_COUNT; ++word)
    {
        meta_ptr[word] = PERF_COMPRESSED_META_WORDS[word];
    }
#else
    (void)meta_base;
#endif
}

// CUSTOM_MM_LAYOUT picks where the output tiles sit in DEST, the same on the math and pack threads. Compressed custom_mm
// has no SplitAcc layouts, so nothing finalizes.
static_assert(CUSTOM_MM_LAYOUT <= static_cast<std::uint8_t>(ckernel::CustomMmDestLayout::Compact), "CUSTOM_MM_LAYOUT is not a CustomMmDestLayout");
constexpr auto dest_layout = static_cast<ckernel::CustomMmDestLayout>(CUSTOM_MM_LAYOUT);

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_compressed_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t in1_face_r_dim     = params.in1_face_r_dim;
    const std::uint32_t in0_face_r_dim     = params.in0_face_r_dim;
    const std::uint32_t num_faces_B        = params.num_faces_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const auto& buffer_C                   = params.buffer_C;
    const auto& buffer_B                   = params.buffer_B;
    const auto& buffer_A                   = params.buffer_A;
#endif

    initialize_perf_compressed_meta(buffer_C[0]);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            in1_face_r_dim,
            in0_face_r_dim,
            num_faces_B,
            num_faces_A,
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);

        if constexpr (num_calls > 1)
        {
            // Race and rearm cases only; the boundary cases' bfp2 stalls clear SrcA to 0 anyway. Leave a non-zero SrcA
            // clear value on unpacker 0: a both-bank SrcB clear only corrupts the SrcA writes it overlaps while unpacker
            // 0's last SrcA clear value is non-zero, and that value outlives kernels. Without this the race would pass
            // or fail depending on what last ran on the core.
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
        }

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
            compressed_mm_isolate_dvalids<true>(LOOP_FACTOR, num_calls, kt_per_call, CT_DIM, buffer_C[0], meta_bytes_per_call);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // An in0 tile is 2 faces x in0_face_r_dim rows x 16 Float16_b datums = 4 * in0_face_r_dim 16-byte words.
                for (std::uint32_t call = 0; call < num_calls; call++)
                {
                    _llk_unpack_AB_compressed_custom_mm_(
                        L1_ADDRESS(buffer_B[0]) + call * b_words_per_call,
                        L1_ADDRESS(buffer_A[0]) + call * kt_per_call * 4 * in0_face_r_dim,
                        buffer_C[0] + call * meta_bytes_per_call,
                        kt_per_call,
                        CT_DIM);
                    if constexpr (CUSTOM_MM_REARM)
                    {
                        // Rearm cases only. Leave a -inf SrcA clear value after every call, as a max-reduce or top-k between
                        // calls would. Each call ends on the end-of-call stall, a SrcA clear to 0, which would otherwise hide a
                        // both-bank clear in the execute at every boundary after the first. Like the end stall it waits for its
                        // bank, so it never clears a bank math still reads.
                        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1 /* Stall_Clr_Cntrl */, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_compressed_custom_mm.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& buffer_C            = params.buffer_C;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        _llk_math_compressed_custom_mm_init_<false /* transpose */, dest_layout>();

        LLK_ASSERT(
            (custom_mm_fits_dest<dest_sync, dest_layout>(0 /* dst_index */, CT_DIM)), "compressed_custom_mm: CT_DIM output tiles overrun the DEST sync region");
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            compressed_mm_isolate_dvalids<false>(LOOP_FACTOR, num_calls, kt_per_call, CT_DIM, buffer_C[0], meta_bytes_per_call);
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
                    if constexpr (num_calls > 1)
                    {
#ifndef LLK_PROFILER
                        // Functional race cases stagger math so unpack reaches the next call boundary.
                        ckernel::wait(2000 /* cycles */);
#endif
                    }
                    _llk_math_compressed_custom_mm_<dest_layout>(buffer_C[0] + call * meta_bytes_per_call, 0 /* dst_index */, kt_per_call, CT_DIM);
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
#include "params.h"

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

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_face_r_dim, TILE_C_DIM, num_faces, true);

        _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
            formats.pack_src, in0_face_r_dim, TILE_C_DIM, num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
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

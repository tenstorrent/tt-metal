// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t num_calls = CUSTOM_MM_NUM_CALLS;

static constexpr std::uint32_t METAS_PER_WORD   = 5;
static constexpr std::uint32_t META_STRIDE      = 6;
static constexpr std::uint32_t META_HEADER_BITS = 2;
static constexpr std::uint32_t FACES_PER_META   = 4;
static constexpr std::uint32_t META_MASK        = (1u << META_STRIDE) - 1;
static constexpr std::uint32_t FACE_BITS_MASK   = (1u << FACES_PER_META) - 1;
static constexpr std::uint32_t METAS_PER_BLOCK  = 2 * CT_DIM;
static constexpr std::uint32_t NUM_BLOCKS       = KT_DIM / 2;

// Copy the compile-time metadata into the reserved perf buffer.
inline void initialize_perf_compressed_meta(const std::uint32_t meta_base)
{
#ifdef LLK_PROFILER
    constexpr std::uint32_t math_words = (KT_DIM * CT_DIM + METAS_PER_WORD - 1) / METAS_PER_WORD;
    static_assert(PERF_COMPRESSED_META_WORD_COUNT > math_words, "perf metadata is missing the math/iters words");
    static_assert(
        PERF_COMPRESSED_META_WORD_COUNT ==
            math_words + 1 + 2 * ((PERF_COMPRESSED_META_WORDS[math_words] / 6) / 8 + 1) + (PERF_COMPRESSED_META_WORDS[math_words] + 5) / 6,
        "perf metadata size does not match the face-compressed layout");
    volatile std::uint32_t* meta_ptr = reinterpret_cast<volatile std::uint32_t*>(meta_base);
    for (std::uint32_t word = 0; word < PERF_COMPRESSED_META_WORD_COUNT; ++word)
    {
        meta_ptr[word] = PERF_COMPRESSED_META_WORDS[word];
    }
#else
    (void)meta_base;
#endif
}

struct FaceCompressedBurst
{
    std::uint32_t nonzero_faces;
    bool last_nonzero;
};

#ifdef LLK_PROFILER
constexpr std::array<FaceCompressedBurst, NUM_BLOCKS> make_face_compressed_bursts()
{
    std::array<FaceCompressedBurst, NUM_BLOCKS> bursts {};
    for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
    {
        for (std::uint32_t m = 0; m < METAS_PER_BLOCK; ++m)
        {
            const std::uint32_t meta_index = block * METAS_PER_BLOCK + m;
            const std::uint32_t meta = (PERF_COMPRESSED_META_WORDS[meta_index / METAS_PER_WORD] >> (META_STRIDE * (meta_index % METAS_PER_WORD))) & META_MASK;
            const std::uint32_t face_bits = (meta >> META_HEADER_BITS) & FACE_BITS_MASK;
            for (std::uint32_t face = 0; face < FACES_PER_META; ++face)
            {
                const bool nonzero = (face_bits & (1u << face)) != 0;
                bursts[block].nonzero_faces += nonzero;
                if (m + 1 == METAS_PER_BLOCK && face + 1 == FACES_PER_META)
                {
                    bursts[block].last_nonzero = nonzero;
                }
            }
        }
    }
    return bursts;
}

static constexpr auto FACE_COMPRESSED_BURSTS = make_face_compressed_bursts();
#endif

// Count-equivalent approximation: coalesce each block into bursts instead of
// reproducing the unpacker's up-to-three-face SrcB lookahead.
template <bool post>
inline void face_compressed_isolate_dvalids(const std::uint32_t loop_factor)
{
#ifdef LLK_PROFILER
    for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
    {
        for (const FaceCompressedBurst burst : FACE_COMPRESSED_BURSTS)
        {
            if constexpr (post)
            {
                _perf_unpack_loop_set_valid<false /* set_a */, true /* set_b */>(1 /* iterations */);
                if (burst.nonzero_faces > 0)
                {
                    _perf_unpack_loop_set_valid<true /* set_a */, false /* set_b */>(burst.nonzero_faces);
                }
            }
            else if (burst.last_nonzero)
            {
                if (burst.nonzero_faces > 1)
                {
                    _perf_math_loop_clear_valid<true /* clear_a */, false /* clear_b */>(burst.nonzero_faces - 1);
                }
                _perf_math_loop_clear_valid<true /* clear_a */, true /* clear_b */>(1 /* iterations */);
            }
            else
            {
                if (burst.nonzero_faces > 0)
                {
                    _perf_math_loop_clear_valid<true /* clear_a */, false /* clear_b */>(burst.nonzero_faces);
                }
                _perf_math_loop_clear_valid<false /* clear_a */, true /* clear_b */>(1 /* iterations */);
            }
        }
        if constexpr (CT_DIM == 1)
        {
            if constexpr (post)
            {
                _perf_unpack_loop_set_valid<false /* set_a */, true /* set_b */>(1 /* iterations */);
                _perf_unpack_loop_set_valid<true /* set_a */, false /* set_b */>(1 /* iterations */);
            }
            else
            {
                _perf_math_loop_clear_valid<true /* clear_a */, true /* clear_b */>(1 /* iterations */);
            }
        }
    }
#else
    (void)loop_factor;
#endif
}

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_face_compressed_mm.h"
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

        _llk_unpack_AB_face_compressed_mm_init_<false>(in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            face_compressed_isolate_dvalids<true>(LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t call = 0; call < num_calls; ++call)
                {
                    _llk_unpack_AB_face_compressed_mm_<CT_DIM, true, true>(L1_ADDRESS(buffer_A[0]), buffer_C[0], KT_DIM);
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_unpack_AB_face_compressed_mm_uninit_(num_faces_B);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_face_compressed_mm.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t in0_face_r_dim = params.in0_face_r_dim;
    const auto& buffer_C               = params.buffer_C;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        _llk_math_face_compressed_mm_init_<CT_DIM>(in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            face_compressed_isolate_dvalids<false>(LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                for (std::uint32_t call = 0; call < num_calls; ++call)
                {
                    _llk_math_face_compressed_mm_<CT_DIM, true>(buffer_C[0], in0_face_r_dim, 0, KT_DIM);
                }
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
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_face_r_dim, TILE_C_DIM, num_faces, true);

        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/, true /*mutex_ADC*/>(
            formats.pack_src, in0_face_r_dim, TILE_C_DIM, num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
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
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default, true /* mutex_ADC */>(i, L1_ADDRESS(buffer_Res[i]));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif

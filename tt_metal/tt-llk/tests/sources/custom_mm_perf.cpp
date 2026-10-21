// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf kernel for the plain custom_mm (llk_{unpack_AB,math,pack}_custom_mm.h). Each loop runs one
// KT_DIM x CT_DIM block, split over CUSTOM_MM_NUM_CALLS back-to-back calls that accumulate into one DEST.
//
// CUSTOM_MM_SPLIT_ACC   accumulate split partials and finalize them on the last call
// CUSTOM_MM_DENSE       tiles 32 DEST rows apart
// CUSTOM_MM_COMPACT     tiles 16 DEST rows apart, output face 1 at row 8
// which together pick the CustomMmDestLayout.
//
// Operand values do not matter for timing, so the in1 walk (KT_DIM * CT_DIM contiguous tiles) is allowed
// to read past the stimuli ring.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
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

constexpr bool split_acc = CUSTOM_MM_SPLIT_ACC;
static_assert(!(CUSTOM_MM_COMPACT && (CUSTOM_MM_DENSE || CUSTOM_MM_SPLIT_ACC)), "Compact has no dense or split_acc variant");
constexpr ckernel::CustomMmDestLayout dest_layout = CUSTOM_MM_COMPACT ? ckernel::CustomMmDestLayout::Compact
                                                    : CUSTOM_MM_DENSE
                                                        ? (split_acc ? ckernel::CustomMmDestLayout::DenseSplitAcc : ckernel::CustomMmDestLayout::Dense)
                                                        : (split_acc ? ckernel::CustomMmDestLayout::DefaultSplitAcc : ckernel::CustomMmDestLayout::Default);
constexpr std::uint32_t num_calls                 = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call               = KT_DIM / num_calls;
static_assert(kt_per_call * num_calls == KT_DIM, "K tiles must split evenly across the calls");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        // in1 (full tiles) is SrcA, in0 ([M,32] tiles) is SrcB, as in custom_mm_test.cpp
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
        _llk_unpack_AB_custom_mm_init_<false /* transpose */, true /* clear_src */>(params.in0_face_r_dim, formats.unpack_B_dst, CT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Per k-tile: one SrcB valid, then CT_DIM SrcA valids
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                for (std::uint32_t k = 0; k < KT_DIM; k++)
                {
                    _perf_unpack_set_valid(ckernel::SrcB);
                    for (std::uint32_t c = 0; c < CT_DIM; c++)
                    {
                        _perf_unpack_set_valid(ckernel::SrcA);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                for (std::uint32_t call = 0; call < num_calls; call++)
                {
                    _llk_unpack_AB_custom_mm_<false /* read_transposed */>(
                        PERF_ADDRESS(PERF_INPUT_B, 0),
                        PERF_ADDRESS(PERF_INPUT_A, 0),
                        call * kt_per_call * CT_DIM /* tile_index_a */,
                        0 /* tile_index_b */,
                        params.TILE_SIZE_UNPACK_B,
                        params.TILE_SIZE_UNPACK_A,
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

template <bool last_call>
inline void run_call(const std::uint32_t in0_face_r_dim)
{
    _llk_math_custom_mm_<split_acc && last_call /* finalize */, dest_layout>(in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM);
}

inline void run_block(const std::uint32_t in0_face_r_dim)
{
    for (std::uint32_t call = 0; call + 1 < num_calls; call++)
    {
        run_call<false>(in0_face_r_dim);
    }
    run_call<true>(in0_face_r_dim);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_custom_mm_init_<false /* transpose */, dest_layout>(params.in0_face_r_dim, CT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // Drain what the unpacker produces: per k-tile CT_DIM SrcA valids, then the SrcB one
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                for (std::uint32_t k = 0; k < KT_DIM; k++)
                {
                    for (std::uint32_t c = 0; c < CT_DIM; c++)
                    {
                        _perf_math_clear_valid(ckernel::SrcA);
                    }
                    _perf_math_clear_valid(ckernel::SrcB);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // No dest sync: pack returns immediately
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                run_block(params.in0_face_r_dim);
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                run_block(params.in0_face_r_dim);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_custom_mm.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);
        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
            formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        if constexpr (dest_layout == CustomMmDestLayout::Compact)
        {
            _llk_pack_custom_mm_init_<dest_layout, is_fp32_dest_acc_en>();
        }
        else if constexpr (custom_mm_tile_rows(dest_layout) == 32)
        {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * (is_fp32_dest_acc_en ? 4 : 2));
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                // Kept rolled, as with a runtime tile count. GCC fully unrolls up to 16 constant iterations,
                // which packs ~4.5 cycles per tile faster and would skew narrow widths against wide ones
#pragma GCC unroll 1
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(LOOP_FACTOR); loop++)
            {
                _llk_packer_wait_for_math_done_();
#pragma GCC unroll 1
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

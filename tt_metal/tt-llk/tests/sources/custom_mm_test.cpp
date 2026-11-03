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
        // in1 (B, full [32,32] tiles) is SrcA; in0 (A, [M,32] tiles) is SrcB.
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
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
        }

        _llk_unpack_AB_custom_mm_init_<false /* transpose */, true /* clear_src */>(params.in0_face_r_dim, formats.unpack_B_dst, CT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Per k-tile the MOP posts one SrcA+SrcB, then CT_DIM-1 SrcA-only
            // reuses. Posting every SrcA first fills the depth-2 bank while math
            // is still waiting on SrcB.
            for (std::uint32_t k = 0; k < LOOP_FACTOR * num_calls * kt_per_call; ++k)
            {
                _perf_unpack_loop_set_valid<true, true>(1);
                if constexpr (CT_DIM > 1)
                {
                    _perf_unpack_loop_set_valid<true, false>(CT_DIM - 1);
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        if constexpr (CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG)
        {
            math::_configure_preserve_zero_flag_state_();
        }

        _llk_math_custom_mm_init_<false /* transpose */, dest_layout>(params.in0_face_r_dim, CT_DIM);

        LLK_ASSERT((custom_mm_fits_dest<dest_sync, dest_layout>(0 /* dst_index */, CT_DIM)), "custom_mm: CT_DIM output tiles overrun the DEST sync region");
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // Math clears SrcA on the first CT_DIM-1 tiles and SrcA+SrcB on the last.
            for (std::uint32_t k = 0; k < LOOP_FACTOR * num_calls * kt_per_call; ++k)
            {
                if constexpr (CT_DIM > 1)
                {
                    _perf_math_loop_clear_valid<true, false>(CT_DIM - 1);
                }
                _perf_math_loop_clear_valid<true, true>(1);
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
                    if constexpr (num_calls > 1)
                    {
                        ckernel::wait(2000);
                    }
                    // The last call of the iteration must run the layout's default
                    // finalize. SplitAcc merges partials there; every layout's MOP
                    // ends that call with CLR_AB, so the next LOOP_FACTOR iteration
                    // does not inherit a live Src bank. Earlier calls only accumulate.
                    if (call == num_calls - 1)
                    {
                        _llk_math_custom_mm_<dest_layout>(params.in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM);
                    }
                    else
                    {
                        _llk_math_custom_mm_<dest_layout, false /* finalize */>(params.in0_face_r_dim, 0 /* dst_index */, kt_per_call, CT_DIM);
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
            formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
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
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
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

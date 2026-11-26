// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Driver for calculate_reshuffle_rows (llk_sfpu/ckernel_sfpu_reshuffle_rows.h), functional and perf.
//
// buffer_A holds (input, accumulator) tile pairs, all copied to Dest in one section; as embedding_backward
// does per pair, math copies both tiles, runs the init and then the kernel at Dest tile 2k, which
// accumulates into tile 2k + 1. buffer_B holds the 32-byte row mask.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_stream_set_valid<true /* set_a */, is_fp32_dest_acc_en /* set_b */, PERF_RUN_TYPE>(
                    TILE_NUM_FACES * TILE_CNT * LOOP_FACTOR /* iterations */);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

// The macros header declares reshuffle_rows_init, so it goes before the kernel header that defines it
// (-Wredundant-decls otherwise).
// clang-format off
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_sfpu/ckernel_sfpu_reshuffle_rows.h"
// clang-format on

static constexpr std::uint32_t MAX_TILES_DEST = ckernel::get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();
static_assert(TILE_CNT % 2 == 0 && TILE_CNT <= MAX_TILES_DEST, "tile pairs must fit one Dest section");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_B = params.buffer_B;
#endif
    // The kernel skips a 16-byte tile header before the mask.
    const auto idx_addr = buffer_B[0] - 16;

    auto run_pairs = [&]()
    {
        for (std::uint32_t tile = 0; tile < TILE_CNT; tile += 2)
        {
            if constexpr (!unpack_to_dest || PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile /* dst_index */, formats.math, formats.math);
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile + 1 /* dst_index */, formats.math, formats.math);
            }
            llk_math_eltwise_unary_sfpu_init<SfpuType::reshuffle_rows, is_fp32_dest_acc_en>();
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_reshuffle_rows, (APPROX_MODE), tile /* dst_index */, VectorMode::RC_custom, idx_addr);
        }
    };

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        // Wormhole: one MOP stream (perf.h)
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && perf_mock_mop<PERF_RUN_TYPE>() && !unpack_to_dest)
        {
            _perf_math_stream_clear_valid<true, true, PERF_RUN_TYPE>(TILE_NUM_FACES * TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            tile, formats.math, formats.math);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ true>(TILE_NUM_FACES);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                run_pairs();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                run_pairs();
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
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
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile /* tile_index */, L1_ADDRESS(buffer_Res[tile]));
                }
                // Wormhole: release the dest section as production's tile_regs_release does
                if constexpr (PERF_PACK_SECTION_RELEASE)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile /* tile_index */, L1_ADDRESS(buffer_Res[tile]));
                }
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

// Only modes supported are L1_TO_L1, PACK_ISOLATE and L1_CONGESTION
static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for this benchmark");
static_assert(PERF_RUN_TYPE != PerfRunType::UNPACK_ISOLATE, "Unpack isolation not supported for this benchmark");

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

// Algorithm invariants
static_assert(BLOCK_CT_DIM <= MAX_TILES_DEST, "Block must fit in Dest register");
static_assert(FULL_CT_DIM % BLOCK_CT_DIM == 0, "FULL_CT_DIM must be divisible by BLOCK_CT_DIM");

// Test assumptions
// static_assert(FULL_RT_DIM * FULL_CT_DIM == TILE_CNT, "FULL_RT_DIM * FULL_CT_DIM must be equal to TILE_CNT");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            4 /* num_faces */,
            4 /* num_faces */);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    PERF_ADDRESS(PERF_INPUT_A, i), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    constexpr bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            4 /* num_faces */, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reconfig_remap_wrapper_(true);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT * TILE_NUM_FACES);
                return;
            }

            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; block++)
                {
                    for (std::uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                        // "consumed" and can be overwritten with new data.
                        // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                        // this method will perform synchronization only and no actual data copy.
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                    }
                }
            }
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; block++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (std::uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; block_tile++)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        block_tile, formats.math, formats.math);
                }
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    constexpr bool UNTILIZE = true;

    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        _llk_pack_untilize_init_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4 /* num_faces */);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile += BLOCK_CT_DIM)
                {
                    _llk_pack_untilize_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM>(
                        PERF_ADDRESS(PERF_OUTPUT, tile), formats.pack_dst, FACE_R_DIM, 4 /* num_faces */, 0 /* tile_dst_rt_offset */);
                }
            }
            PROFILER_SYNC();
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; i += BLOCK_CT_DIM)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_untilize_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM>(
                    PERF_ADDRESS(PERF_OUTPUT, i), formats.pack_dst, FACE_R_DIM, 4 /* num_faces */, 0 /* tile_dst_rt_offset */);
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

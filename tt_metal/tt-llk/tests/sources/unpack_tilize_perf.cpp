// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "build.h"
#include "ckernel.h"
#include "counters.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_tilize");

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include <algorithm>

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT     = params.TILE_CNT;
    const std::uint32_t FULL_CT_DIM  = params.FULL_CT_DIM;
    const std::uint32_t FULL_RT_DIM  = params.FULL_RT_DIM;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
#endif
    LLK_ASSERT(FULL_RT_DIM * FULL_CT_DIM == TILE_CNT, "FULL_RT_DIM * FULL_CT_DIM must be equal to TILE_CNT");
    constexpr std::uint32_t src = 0x65000;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            4 /* num_faces */,
            4 /* num_faces */);
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
            {
                const std::uint32_t tile_row_addr = L1_ADDRESS(src + (i % 8) * 0x1000); // TODO SS<-LP use PERF_ADDRESS here
                for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                {
                    _llk_unpack_tilize_wrapper_(
                        tile_row_addr,
                        j,
                        formats.unpack_A_src,
                        formats.unpack_A_dst,
                        0 /* block_ct_dim */,
                        FACE_R_DIM,
                        4 /* num_faces */,
                        false /* narrow_tile */);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

static constexpr bool TILIZE = true;

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"

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
    const bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")
        // copy srca to dest
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            is_int_fpu_en,
            llk_test_pack_mode_v<false, TILIZE>>(4 /* num_faces */, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
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
            const std::uint32_t NUM_DVALIDS = _llk_unpack_tilize_num_dvalids_wrapper_(TILE_CNT, TILE_NUM_FACES);
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * NUM_DVALIDS);
                return;
            }

            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; i++)
                {
                    LLK_ASSERT(
                        (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");

                    // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                    // "consumed" and can be overwritten with new data.
                    // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                    // this method will perform synchronization only and no actual data copy.
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i, formats.math, formats.math);
                }
            }
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            std::uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                std::uint32_t num_tiles = std::min(remaining_tiles, MAX_TILES_DEST);
                for (std::uint32_t i = 0; i < num_tiles; ++i)
                {
                    LLK_ASSERT(
                        (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i, formats.math, formats.math);
                }
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                remaining_tiles -= num_tiles;
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include <algorithm>

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
    static constexpr bool UNTILIZE = false;

    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
        _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    const std::uint32_t tile_index = i % MAX_TILES_DEST;
                    LLK_ASSERT(
                        (tile_index < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
                }
            }
            PROFILER_SYNC();
            return;
        }

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            std::uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                std::uint32_t num_tiles = std::min(remaining_tiles, MAX_TILES_DEST);
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t i = 0; i < num_tiles; ++i)
                {
                    const std::uint32_t tile_index = i % MAX_TILES_DEST;
                    LLK_ASSERT(
                        (tile_index < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                remaining_tiles -= num_tiles;
            }
        }
        PROFILER_SYNC();
    }
}

#endif

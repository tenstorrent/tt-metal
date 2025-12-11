// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

static constexpr uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_src,
            formats.unpack_src,
            formats.unpack_dst,
            formats.unpack_dst,
            FACE_R_DIM,
            /* within_face_16x16_transpose */ 0,
            /* num_faces */ 4);
        _llk_unpack_AB_init_<>(
            FACE_R_DIM,
            TILE_NUM_FACES,
            /* narrow tile */ false,
            /* transpose within face */ 0);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }
        else
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<>(formats.math, formats.math);
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(TILE_NUM_FACES, false);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            return _perf_math_loop_clear_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                for (uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                        TILE_NUM_FACES, block_tile, false);
                }
            }
        }
        else
        {
            for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                        TILE_NUM_FACES, block_tile, false);
                }
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                for (uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                }
            }
        }
        else
        {
            for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_packer_wait_for_math_done_();
                for (uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

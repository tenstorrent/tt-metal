// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "build.h"
#include "ckernel.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// Invariants for the test to run correctly
static_assert(BLOCK_RT_DIM * BLOCK_CT_DIM == TILE_CNT, "BLOCK_RT_DIM * BLOCK_CT_DIM must be equal to TILE_CNT");

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_tilize");

static constexpr uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include <algorithm>

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel()
{
    constexpr uint32_t src = 0x1A000;
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_dst, FACE_R_DIM, 0, 4);
        _llk_unpack_tilize_init_(formats.unpack_src, formats.unpack_dst, BLOCK_CT_DIM, FACE_R_DIM, false);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)
            {
                const uint32_t tile_row_addr = L1_ADDRESS(src + (i % 8) * 0x1000); // TODO SS<-LP use PERF_ADDRESS here
                for (uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                {
                    _llk_unpack_tilize_(tile_row_addr, j, formats.unpack_src, 0, FACE_R_DIM, 4, false);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

const bool TILIZE = true;

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

void run_kernel()
{
    const bool is_int_fpu_en = false;

    {
        ZONE_SCOPED("INIT")
        // copy srca to dest
#ifdef ARCH_BLACKHOLE
        // set tilize flag to true
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, TILIZE, is_int_fpu_en>(4, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<false, false>(formats.math, formats.math);
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
#ifdef ARCH_BLACKHOLE
            // Due to the blackhole tilize bug mitigation
            // DVALID is set for each tile, instead of each face.
            constexpr uint32_t NUM_DVALIDS = TILE_CNT;
#else
            constexpr uint32_t NUM_DVALIDS = TILE_CNT * TILE_NUM_FACES;
#endif
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * NUM_DVALIDS);
                return;
            }

            for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (uint32_t i = 0; i < TILE_CNT; i++)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i, formats.math, formats.math);
                }
            }
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                uint32_t num_tiles = std::min(remaining_tiles, MAX_TILES_DEST);
                for (uint32_t i = 0; i < num_tiles; ++i)
                {
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

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    constexpr uint32_t dst = 0x1E000;
    const bool UNTILIZE    = false;

    {
        ZONE_SCOPED("INIT")

#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, TILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false, TILIZE>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
#endif
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(i, L1_ADDRESS(dst + (i % 8) * 0x1000)); // TODO SS<-LP use PERF_ADDRESS here
                }
            }
            PROFILER_SYNC();
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                uint32_t num_tiles = std::min(remaining_tiles, MAX_TILES_DEST);
                _llk_packer_wait_for_math_done_();
                for (uint32_t i = 0; i < num_tiles; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(i, L1_ADDRESS(dst + (i % 8) * 0x1000)); // TODO SS<-LP use PERF_ADDRESS here
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                remaining_tiles -= num_tiles;
            }
        }
        PROFILER_SYNC();
    }
}

#endif

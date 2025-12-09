// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// Only modes supported are L1_TO_L1, PACK_ISOLATE and L1_CONGESTION
static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for this benchmark");
static_assert(PERF_RUN_TYPE != PerfRunType::UNPACK_ISOLATE, "Unpack isolation not supported for this benchmark");

static constexpr uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

// Algorithm invariants
static_assert(BLOCK_CT_DIM <= MAX_TILES_DEST, "Block must fit in Dest register");
static_assert(FULL_CT_DIM % BLOCK_CT_DIM == 0, "FULL_CT_DIM must be divisible by BLOCK_CT_DIM");

// Test assumptions
static_assert(FULL_RT_DIM * FULL_CT_DIM == TILE_CNT, "FULL_RT_DIM * FULL_CT_DIM must be equal to TILE_CNT");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);
        _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_dst, FACE_R_DIM, 0, 4);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        if (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (int i = 0; i < TILE_CNT; ++i)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    PERF_ADDRESS(PERF_INPUT_A, i), formats.unpack_src, formats.unpack_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

void run_kernel()
{
    constexpr bool is_int_fpu_en = false;

    {
        ZONE_SCOPED("INIT")

#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(4, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<true, false>(formats.math, formats.math);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

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

            // FIXME: Currently have no way to mock math for unpack to dest
            for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; block++)
                {
                    for (uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; block_tile++)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                    }
                }
            }
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; block++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; block_tile++)
                {
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

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    constexpr bool UNTILIZE = true;

    {
        ZONE_SCOPED("INIT")

#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
        _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4);
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
        _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_dst, FACE_R_DIM, 4);
#endif
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (uint32_t tile = 0; tile < TILE_CNT; tile += BLOCK_CT_DIM)
                {
                    _llk_pack_untilize_<BLOCK_CT_DIM, FULL_CT_DIM>(PERF_ADDRESS(PERF_OUTPUT, tile), formats.pack_dst, FACE_R_DIM, 4, 0);
                }
            }
            PROFILER_SYNC();
            return;
        }

        for (uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (uint32_t i = 0; i < TILE_CNT; i += BLOCK_CT_DIM)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_untilize_<BLOCK_CT_DIM, FULL_CT_DIM>(PERF_ADDRESS(PERF_OUTPUT, i), formats.pack_dst, FACE_R_DIM, 4, 0);
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The production shapes of a reduce over a block of tiles: REDUCE_ROW and REDUCE_SCALAR accumulate eight input tiles into
// one output tile (a row of eight tiles), REDUCE_COL writes one output tile per input tile (a DEST section of columns).
// The input tiles go through calls of REDUCE_BLOCK_CT_DIM tiles: 1 is the per tile call, more is one
// _llk_unpack_AB_reduce_block_ and one _llk_math_reduce_block_ per call. TILE_CNT counts input tiles; the data tiles sit at
// the page size of their format (TILE_SIZE_UNPACK_A), as in a circular buffer.
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr std::uint32_t MAX_TILES_DEST    = is_fp32_dest_acc_en ? 4 : 8;
static constexpr std::uint32_t CT                = REDUCE_BLOCK_CT_DIM;
static constexpr bool ACCUMULATE                 = REDUCE_DIM != ckernel::ReduceDim::REDUCE_COL;
static constexpr std::uint32_t TILES_PER_OUTPUT  = ACCUMULATE ? 8 : 1;
static constexpr std::uint32_t TILES_PER_SECTION = MAX_TILES_DEST * TILES_PER_OUTPUT;
static constexpr std::uint32_t DST_STRIDE        = ACCUMULATE ? 0 : 1;
static_assert(TILES_PER_OUTPUT % CT == 0 || !ACCUMULATE, "a call must not cross an output tile");

static constexpr bool IS_FULL_TILE_REDUCE_ROW   = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) && (POOL_TYPE != ckernel::PoolType::MAX);
static constexpr std::uint32_t DVALIDS_PER_TILE = IS_FULL_TILE_REDUCE_ROW ? 1 : TILE_NUM_FACES;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t TILE_STRIDE = params.TILE_SIZE_UNPACK_A;
#else
    constexpr std::uint32_t TILE_STRIDE = TILE_SIZE_UNPACK_A;
#endif
    constexpr bool holds_scaler = reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(DEFAULT_TENSOR_SHAPE);
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            /* num_faces */ 4,
            /* num_faces */ 4);
        _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(DEFAULT_TENSOR_SHAPE);
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
            for (std::uint32_t section = 0; section < TILE_CNT; section += TILES_PER_SECTION)
            {
                const std::uint32_t section_end = std::min(TILE_CNT, section + TILES_PER_SECTION);
                for (std::uint32_t call = section; call < section_end; call += CT)
                {
                    const std::uint32_t n = std::min(CT, section_end - call);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
                    {
                        if (holds_scaler && n > 1)
                        {
                            _perf_unpack_set_valid(ckernel::SrcA);
                            for (std::uint32_t t = 0; t < n; t++)
                            {
                                _perf_unpack_set_valid(ckernel::SrcB);
                            }
                        }
                        else
                        {
                            _perf_unpack_loop_set_valid<true, true>(n * DVALIDS_PER_TILE);
                        }
                    }
                    else
                    {
                        _llk_unpack_AB_reduce_block_<POOL_TYPE, REDUCE_DIM>(
                            PERF_ADDRESS(PERF_INPUT_A, 0) + call * TILE_STRIDE,
                            PERF_ADDRESS(PERF_INPUT_B, 0),
                            n,
                            TILE_STRIDE,
                            formats.unpack_A_src,
                            DEFAULT_TENSOR_SHAPE);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    constexpr bool holds_scaler = reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(DEFAULT_TENSOR_SHAPE);
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(DEFAULT_TENSOR_SHAPE);
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
            for (std::uint32_t section = 0; section < TILE_CNT; section += TILES_PER_SECTION)
            {
                const std::uint32_t section_end = std::min(TILE_CNT, section + TILES_PER_SECTION);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                for (std::uint32_t call = section; call < section_end; call += CT)
                {
                    const std::uint32_t n = std::min(CT, section_end - call);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
                    {
                        if (holds_scaler && n > 1)
                        {
                            for (std::uint32_t t = 0; t < n; t++)
                            {
                                _perf_math_clear_valid(ckernel::SrcB);
                            }
                            _perf_math_clear_valid(ckernel::SrcA);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<true, true>(n * DVALIDS_PER_TILE);
                        }
                    }
                    else
                    {
                        _llk_math_reduce_block_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(
                            (call - section) / TILES_PER_OUTPUT, n, DST_STRIDE, DEFAULT_TENSOR_SHAPE);
                    }
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
#include "llk_pack.h"
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
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_reduce_mask_config_<REDUCE_DIM>();
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_pack_reduce_mask_clear_();
            return;
        }
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t section = 0; section < TILE_CNT; section += TILES_PER_SECTION)
            {
                const std::uint32_t outputs = (std::min(TILE_CNT, section + TILES_PER_SECTION) - section) / TILES_PER_OUTPUT;
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t tile = 0; tile < outputs; tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, section / TILES_PER_OUTPUT + tile));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}

#endif

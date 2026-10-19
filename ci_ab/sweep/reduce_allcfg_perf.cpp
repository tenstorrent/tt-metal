// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Scratch all-configuration reduce perf kernel (not for merge). Axes beyond perf_reduce / perf_reduce_block:
// tile shape (CI_FACE_R, CI_NF_R, CI_NF_C), DEST sync (CI_SYNC_FULL), call form (CI_CT: 0 = one call per tile,
// n = block calls of n tiles; per-tile calls on main, where the block call does not exist), inputs per output
// (CI_K: 1, or 8 accumulated into one DEST tile with stride 0). CI_REDUCE_HEAD comes from ci_reduce_side.h.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ci_reduce_side.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr DstSync DST_SYNC                = CI_SYNC_FULL ? DstSync::SyncFull : DstSync::SyncHalf;
static constexpr std::uint32_t MAX_TILES_DEST    = (is_fp32_dest_acc_en ? 4 : 8) * (CI_SYNC_FULL ? 2 : 1);
static constexpr std::uint32_t CT                = CI_CT == 0 ? 1 : CI_CT;
// Side 3 models the compute API, which sends BFP blocks and the HiFi held-scaler row sums tile by tile
static constexpr bool BLOCK_CALL                 = CI_CT != 0 && CI_REDUCE_HEAD && !(CI_REDUCE_HEAD >= 3 && CI_API_TILES);
// Side 4 models the API that runs a SCALAR block's math as a block also when its unpacker goes tile by tile (BFP data)
static constexpr bool MATH_BLOCK_CALL = BLOCK_CALL || (CI_CT != 0 && CI_REDUCE_HEAD == 4 && REDUCE_DIM == ReduceDim::REDUCE_SCALAR);
static constexpr std::uint32_t TILES_PER_OUTPUT  = CI_K;
static constexpr std::uint32_t TILES_PER_SECTION = MAX_TILES_DEST * TILES_PER_OUTPUT;
static constexpr std::uint32_t DST_STRIDE        = CI_K > 1 ? 0 : 1;
static_assert(TILES_PER_OUTPUT % CT == 0 || CI_K == 1, "a call must not cross an output tile");

static constexpr ckernel::TensorShape CI_SHAPE = {
    static_cast<std::uint8_t>(CI_FACE_R), static_cast<std::uint8_t>(FACE_C_DIM), static_cast<std::uint8_t>(CI_NF_R), static_cast<std::uint8_t>(CI_NF_C)};
static constexpr std::uint32_t CI_FACES = CI_NF_R * CI_NF_C;
static constexpr bool CI_FULL_TILE      = CI_FACE_R == FACE_R_DIM && CI_FACES == 4;
static constexpr bool ROW_SUM_FULL      = (REDUCE_DIM == ReduceDim::REDUCE_ROW) && (POOL_TYPE != PoolType::MAX) && CI_FULL_TILE;
static constexpr std::uint32_t DVALIDS  = ROW_SUM_FULL ? 1 : CI_FACES;

static inline std::uint32_t out_dst(std::uint32_t call, std::uint32_t section)
{
    return (call - section) / TILES_PER_OUTPUT;
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    [[maybe_unused]] const std::uint32_t TILE_STRIDE = params.TILE_SIZE_UNPACK_A;
#if CI_REDUCE_HEAD
#if CI_REDUCE_HEAD == 2
    const bool holds_scaler = BLOCK_CALL && reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(CI_SHAPE, formats.unpack_A_src);
#else
    constexpr bool holds_scaler = BLOCK_CALL && reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(CI_SHAPE);
#endif
#else
    constexpr bool holds_scaler = false;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, CI_FACE_R, CI_FACE_R, CI_FACES, CI_FACES);
        _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(CI_SHAPE);
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
                            _perf_unpack_loop_set_valid<true, true>(n * DVALIDS);
                        }
                    }
                    else
                    {
#if CI_REDUCE_HEAD
                        if constexpr (BLOCK_CALL)
                        {
                            _llk_unpack_AB_reduce_block_<POOL_TYPE, REDUCE_DIM>(
                                PERF_ADDRESS(PERF_INPUT_A, 0) + call * TILE_STRIDE, PERF_ADDRESS(PERF_INPUT_B, 0), n, TILE_STRIDE, formats.unpack_A_src, CI_SHAPE);
                        }
                        else
#endif
                        {
                            for (std::uint32_t t = 0; t < n; t++)
                            {
                                _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(PERF_ADDRESS(PERF_INPUT_A, call + t), PERF_ADDRESS(PERF_INPUT_B, 0));
                            }
                        }
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#if CI_REDUCE_HEAD
#if CI_REDUCE_HEAD == 2
    const bool holds_scaler = BLOCK_CALL && reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(CI_SHAPE, formats.unpack_A_src);
#else
    constexpr bool holds_scaler = BLOCK_CALL && reduce_block_holds_scaler<POOL_TYPE, REDUCE_DIM>(CI_SHAPE);
#endif
#else
    constexpr bool holds_scaler = false;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(CI_SHAPE);
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
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
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
                            _perf_math_loop_clear_valid<true, true>(n * DVALIDS);
                        }
                    }
                    else
                    {
#if CI_REDUCE_HEAD
                        if constexpr (MATH_BLOCK_CALL)
                        {
                            _llk_math_reduce_block_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(
                                CI_K > 1 ? out_dst(call, section) : call - section, n, DST_STRIDE, CI_SHAPE
#if CI_REDUCE_HEAD == 2
                                , formats.unpack_A_src
#endif
                            );
                        }
                        else
#endif
                        {
                            for (std::uint32_t t = 0; t < n; t++)
                            {
                                const std::uint32_t dst = CI_K > 1 ? out_dst(call, section) : call - section + t;
                                _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, false>(dst, CI_SHAPE);
                            }
                        }
                    }
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    {
        START_PERF_MEASURE("INIT")
        if constexpr (CI_FULL_TILE)
        {
            _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
            _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst);
            _llk_pack_reduce_mask_config_<REDUCE_DIM>();
            _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        }
        else
        {
            constexpr bool partial_face = CI_FACE_R < FACE_R_DIM;
            constexpr bool narrow_tile  = CI_NF_C == 1;
            _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
                formats.pack_src, formats.pack_dst, CI_SHAPE.total_tensor_size(), CI_FACE_R, CI_SHAPE.total_col_dim(), CI_FACES, partial_face, narrow_tile);
            _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, CI_FACE_R, CI_SHAPE.total_col_dim(), CI_FACES, partial_face, narrow_tile);
            _llk_pack_reduce_mask_config_<REDUCE_DIM>(CI_FACE_R);
            _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(CI_FACE_R, narrow_tile);
        }
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
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, section / TILES_PER_OUTPUT + tile));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}

#endif

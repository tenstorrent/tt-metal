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
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

static constexpr bool IS_REDUCE_ROW = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            /* num_faces */ 4,
            /* num_faces */ 4);
        _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE, /* transpose */ IS_REDUCE_ROW);
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
            return _perf_unpack_loop_set_valid<true, true>(params->TILE_CNT * TILE_NUM_FACES);
        }
        else
        {
            for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile++)
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
#include "llk_math_reduce.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    constexpr MathFidelity MATH_FIDELITY = MathFidelity::HiFi4;

    // todo: INT32 reduce is not supported yet
    constexpr bool ENFORCE_FP32_ACC = false;
    constexpr bool IS_INT_FPU       = false;
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, ENFORCE_FP32_ACC>();
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
            return _perf_math_loop_clear_valid<true, true>(params->TILE_CNT * TILE_NUM_FACES);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "block_tile exceeds max dest tiles");
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, IS_INT_FPU, ENFORCE_FP32_ACC>(block_tile);
                }
            }
        }
        else
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "block_tile exceeds max dest tiles");
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, IS_INT_FPU, ENFORCE_FP32_ACC>(block_tile);
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

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<
            /* untilize */ false,
            /* zero output */ false>(formats.pack_dst);
        _llk_pack_reduce_mask_config_<
            /* untilize */ false,
            REDUCE_DIM>();
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_pack_reduce_mask_clear_();
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "block_tile exceeds max dest tiles");
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                }
            }
        }
        else
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_packer_wait_for_math_done_();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "block_tile exceeds max dest tiles");
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }

        _llk_pack_reduce_mask_clear_();

        PROFILER_SYNC();
    }
}

#endif

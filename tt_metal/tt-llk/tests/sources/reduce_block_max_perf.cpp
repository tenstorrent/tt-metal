// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the experimental block row max (experimental/llk_{math,unpack_AB}_reduce_custom.h) in the shape of reduce_perf.cpp.
// TILE_CNT counts input tiles, as in perf_reduce.py, so the report's cycles per tile are per input tile; the scaler is tile 0 of operand B.
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

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;
static constexpr std::uint32_t CT             = REDUCE_BLOCK_CT_DIM;
static constexpr DstSync DST_SYNC             = DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_reduce_custom.h"
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
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};
    const std::uint32_t num_blocks          = TILE_CNT / CT;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_AB_reduce_block_max_row_init_<CT, is_fp32_dest_acc_en, false>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t i = 0; i < LOOP_FACTOR * num_blocks; i++)
            {
                _perf_unpack_set_valid(ckernel::SrcB);
                for (std::uint32_t t = 0; t < CT; t++)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t b = 0; b < num_blocks; b++)
                {
                    _llk_unpack_AB_reduce_block_max_row_<false>(PERF_ADDRESS(PERF_INPUT_A, b * CT), PERF_ADDRESS(PERF_INPUT_B, 0));
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_AB_reduce_block_max_row_uninit_<false>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_reduce_custom.h"
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
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};
    const std::uint32_t num_blocks          = TILE_CNT / CT;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_block_max_row_init_<CT, is_fp32_dest_acc_en>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t i = 0; i < LOOP_FACTOR * num_blocks; i++)
            {
                for (std::uint32_t t = 0; t < CT; t++)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                }
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t s = 0; s < num_blocks; s += MAX_TILES_DEST)
                {
                    const std::uint32_t n = std::min(num_blocks - s, MAX_TILES_DEST);
                    for (std::uint32_t i = 0; i < n; i++)
                    {
                        _llk_math_reduce_block_max_row_<CT, is_fp32_dest_acc_en>(i, tensor_shape);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t s = 0; s < num_blocks; s += MAX_TILES_DEST)
                {
                    const std::uint32_t n = std::min(num_blocks - s, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                    for (std::uint32_t i = 0; i < n; i++)
                    {
                        _llk_math_reduce_block_max_row_<CT, is_fp32_dest_acc_en>(i, tensor_shape);
                    }
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_reduce_block_max_row_uninit_<is_fp32_dest_acc_en>();
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
    const std::uint32_t num_blocks = TILE_CNT / CT;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_ROW>(FACE_R_DIM);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t s = 0; s < num_blocks; s += MAX_TILES_DEST)
                {
                    const std::uint32_t n = std::min(num_blocks - s, MAX_TILES_DEST);
                    for (std::uint32_t i = 0; i < n; i++)
                    {
                        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en>(i, PERF_ADDRESS(PERF_OUTPUT, s + i));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t s = 0; s < num_blocks; s += MAX_TILES_DEST)
                {
                    const std::uint32_t n = std::min(num_blocks - s, MAX_TILES_DEST);
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t i = 0; i < n; i++)
                    {
                        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en>(i, PERF_ADDRESS(PERF_OUTPUT, s + i));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif

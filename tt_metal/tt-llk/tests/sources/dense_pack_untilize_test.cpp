// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
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
                _perf_unpack_loop_set_valid<true /* src A */, is_fp32_dest_acc_en>(
                    LOOP_FACTOR * params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK * params.num_faces);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < params.NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t input_tile = block * params.NUM_TILES_IN_BLOCK + tile;
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(params.buffer_A[input_tile]), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);
    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            tensor_shape, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reconfig_remap_(true);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block = 0; block < params.NUM_BLOCKS; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math, tensor_shape);
                        }
                    }
                }
            }
            else
            {
                _perf_math_loop_clear_valid<true /* src A */, true /* src B */>(LOOP_FACTOR * params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK * params.num_faces);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block = 0; block < params.NUM_BLOCKS; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math, tensor_shape);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < params.NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            tile, formats.math, formats.math, tensor_shape);
                    }
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, 0 /* tile_size */);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        // Ugly but we are converting one 4 face tile into two 2 face tiles side by side
        _llk_pack_untilize_init_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, TILE_C_DIM, true>(
            formats.pack_src, formats.pack_dst, params.in0_tile_r_dim, params.num_faces / 2);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < params.NUM_BLOCKS; ++block)
                {
                    if constexpr (sync_with_math)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    _llk_pack_untilize_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, 0, true>(
                        L1_ADDRESS(params.buffer_Res[block * params.NUM_TILES_IN_BLOCK]), params.num_faces / 2, 0 /* tile_dst_rt_offset */);
                    if constexpr (sync_with_math)
                    {
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_untilize_uninit_(formats.pack_src);
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

template <typename T>
void unpack_dest_bank_init(const T& params, const FormatConfig& formats, std::uint32_t num_faces)
{
    if constexpr (!tilize_en)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
    }
    else
    {
        _llk_unpack_tilize_init_(formats.unpack_A_src, formats.unpack_A_dst, params.BLOCK_CT_DIM, FACE_R_DIM, false, num_faces);
    }
}

template <typename T>
std::uint32_t unpack_dest_bank_address(const T& params, std::uint32_t tile)
{
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
    {
        return L1_ADDRESS(params.buffer_A[tile]);
    }
    else
    {
        return PERF_ADDRESS(PERF_INPUT_A, tile);
    }
}

template <typename T>
void unpack_dest_bank_tiles(const T& params, const FormatConfig& formats, std::uint32_t num_tiles, std::uint32_t num_faces)
{
    if constexpr (!tilize_en)
    {
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                unpack_dest_bank_address(params, tile), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    else
    {
        for (std::uint32_t row = 0; row < params.BLOCK_RT_DIM; ++row)
        {
            const std::uint32_t read_offset = row * params.BLOCK_RT_DIM;
            for (std::uint32_t column = 0; column < params.BLOCK_CT_DIM; ++column)
            {
                _llk_unpack_tilize_wrapper_(
                    L1_ADDRESS(params.buffer_A[read_offset]),
                    column,
                    formats.unpack_A_src,
                    0 /* unpack_dst_format */,
                    params.BLOCK_CT_DIM,
                    FACE_R_DIM,
                    num_faces,
                    false);
            }
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    const std::uint32_t num_total_tiles = NUM_TILES_IN_BLOCK * NUM_BLOCKS;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        unpack_dest_bank_init(params, formats, num_faces);
        PROFILER_SYNC();

        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_stream_set_valid<true, true, PERF_RUN_TYPE>(TILE_CNT * num_faces * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                unpack_dest_bank_tiles(params, formats, num_total_tiles, num_faces);
            }
            if constexpr (tilize_en)
            {
                _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, num_faces);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces);

    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<
                DataCopyType::A2D,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                is_int_fpu_en,
                llk_test_pack_mode_v<false, tilize_en>>(tensor_shape, formats.math);
            _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        }
        else
        {
            _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
                tensor_shape, formats.math);

            for (std::uint32_t block_tile = 0; block_tile < TILE_CNT; ++block_tile)
            {
                _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    block_tile, formats.math, formats.math, tensor_shape);
            }
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_stream_clear_valid<true, true, PERF_RUN_TYPE>(TILE_CNT * num_faces * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(block_tile, formats.math, formats.math, tensor_shape);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(DST_INDEX + tile, formats.math, formats.math, tensor_shape);
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(block_tile, formats.math, formats.math, tensor_shape);
                    }
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
#include "llk_pack_common.h"

template <typename T>
std::uint32_t pack_dest_bank_address(const T& params, std::uint32_t block, std::uint32_t num_tiles_in_block)
{
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
    {
        return L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block]);
    }
    else
    {
        return PERF_ADDRESS(PERF_OUTPUT, block * num_tiles_in_block);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
    const int L1_ACC                       = params.L1_ACC;
#endif
#ifndef ARCH_BLACKHOLE
    (void)L1_ACC;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(
            formats.pack_src,
            formats.pack_dst,
            FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            NUM_TILES_IN_BLOCK /* num_tiles */);
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            reconfigure_packer_l1_acc(L1_ACC);
        }
        else
        {
            reconfigure_packer_l1_acc(L1_ACC);
            _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_packer_wait_for_math_done_();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                        DST_INDEX, pack_dest_bank_address(params, block, NUM_TILES_IN_BLOCK));
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                        DST_INDEX, pack_dest_bank_address(params, block, NUM_TILES_IN_BLOCK));
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, PERF_ADDRESS(PERF_OUTPUT, block_start));
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
    {
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

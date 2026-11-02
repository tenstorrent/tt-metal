// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

template <typename T, typename = void>
struct has_buffer_a : std::false_type
{
};

template <typename T>
struct has_buffer_a<T, std::void_t<decltype(std::declval<T>().buffer_A)>> : std::true_type
{
};

template <typename T, typename = void>
struct has_buffer_b : std::false_type
{
};

template <typename T>
struct has_buffer_b<T, std::void_t<decltype(std::declval<T>().buffer_B)>> : std::true_type
{
};

template <typename T, typename = void>
struct has_buffer_c : std::false_type
{
};

template <typename T>
struct has_buffer_c<T, std::void_t<decltype(std::declval<T>().buffer_C)>> : std::true_type
{
};

template <typename T>
std::uint32_t ternary_input_a_address(const T& params, std::uint32_t tile)
{
    if constexpr (has_buffer_a<T>::value)
    {
        return L1_ADDRESS(params.buffer_A[tile]);
    }
    else
    {
        return PERF_ADDRESS(PERF_INPUT_A, tile);
    }
}

template <typename T>
std::uint32_t ternary_input_b_address(const T& params, std::uint32_t tile)
{
    if constexpr (has_buffer_b<T>::value)
    {
        return L1_ADDRESS(params.buffer_B[tile]);
    }
    else
    {
        return PERF_ADDRESS(PERF_INPUT_B, tile);
    }
}

template <typename T>
std::uint32_t ternary_input_c_address(const T& params, std::uint32_t tile)
{
    if constexpr (has_buffer_c<T>::value)
    {
        return L1_ADDRESS(params.buffer_C[tile]);
    }
    else
    {
        return PERF_ADDRESS(PERF_INPUT_C, tile);
    }
}

template <typename T, typename = void>
struct has_buffer_res : std::false_type
{
};

template <typename T>
struct has_buffer_res<T, std::void_t<decltype(std::declval<T>().buffer_Res)>> : std::true_type
{
};

template <typename T>
std::uint32_t ternary_output_address(const T& params, std::uint32_t tile)
{
    if constexpr (has_buffer_res<T>::value)
    {
        return L1_ADDRESS(params.buffer_Res[tile]);
    }
    else
    {
        return PERF_ADDRESS(PERF_OUTPUT, tile);
    }
}

template <typename T, typename = void>
struct has_num_blocks : std::false_type
{
};

template <typename T>
struct has_num_blocks<T, std::void_t<decltype(std::declval<T>().NUM_BLOCKS), decltype(std::declval<T>().NUM_TILES_IN_BLOCK)>> : std::true_type
{
};

template <typename T>
std::uint32_t ternary_num_blocks(const T& params)
{
    if constexpr (has_num_blocks<T>::value)
    {
        return params.NUM_BLOCKS;
    }
    else
    {
        return 1;
    }
}

template <typename T>
std::uint32_t ternary_num_tiles_in_block(const T& params)
{
    if constexpr (has_num_blocks<T>::value)
    {
        return params.NUM_TILES_IN_BLOCK;
    }
    else
    {
        return TILE_CNT;
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t NUM_BLOCKS                   = ternary_num_blocks(params);
    const std::uint32_t NUM_TILES_IN_BLOCK           = ternary_num_tiles_in_block(params);
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
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
                _perf_unpack_loop_set_valid<true, is_fp32_dest_acc_en>(num_faces * TILE_CNT * LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t input_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                            ternary_input_a_address(params, input_tile), formats.unpack_A_src, formats.unpack_A_dst);
                        _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                            ternary_input_b_address(params, input_tile), formats.unpack_A_src, formats.unpack_A_dst);
                        _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                            ternary_input_c_address(params, input_tile), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "sfpu_operations.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t NUM_BLOCKS         = ternary_num_blocks(params);
    const std::uint32_t NUM_TILES_IN_BLOCK = ternary_num_tiles_in_block(params);
    constexpr DataCopyType data_copy_type  = DataCopyType::A2D;
    constexpr DataFormat MATH_FORMAT_RAW   = static_cast<DataFormat>(formats.math);
    constexpr DataFormat MATH_FORMAT_ENUM  = (MATH_FORMAT_RAW == DataFormat::Tf32) ? DataFormat::Float32 : MATH_FORMAT_RAW;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        test_utils::call_ternary_sfpu_operation_init<SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid<true, true>(num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<true, true>(num_faces);
                        }
                    }
                }
            }
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
                        if constexpr (!unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        test_utils::call_ternary_sfpu_operation<
                            DST_SYNC_MODE,
                            is_fp32_dest_acc_en,
                            SFPU_TERNARY_OPERATION,
                            APPROX_MODE,
                            is_fp32_dest_acc_en,
                            MATH_FORMAT_ENUM,
                            8>(
                            block_tile, (block_tile + 1) % MAX_TILES_DEST, (block_tile + 2) % MAX_TILES_DEST, block_tile, SFPU_TERNARY_SCALAR, VectorMode::RC);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
                        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(
                            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            0, formats.math, formats.math);
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            1, formats.math, formats.math);
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            2, formats.math, formats.math);
                        _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
#ifdef SFPU_CLOBBER_LREGS
                        ckernel::sfpu::clobber_sfpu_lregs();
#endif
                        test_utils::call_ternary_sfpu_operation_init<SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en>();
                        test_utils::call_ternary_sfpu_operation<
                            DST_SYNC_MODE,
                            is_fp32_dest_acc_en,
                            SFPU_TERNARY_OPERATION,
                            APPROX_MODE,
                            is_fp32_dest_acc_en,
                            MATH_FORMAT_ENUM,
                            8>(0, 1, 2, 0, SFPU_TERNARY_SCALAR, VectorMode::RC);
                        _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                    }
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

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t NUM_BLOCKS         = ternary_num_blocks(params);
    const std::uint32_t NUM_TILES_IN_BLOCK = ternary_num_tiles_in_block(params);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_packer_wait_for_math_done_();
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(0, ternary_output_address(params, result_tile));
                        _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void unpack_dest_bank_init(const FormatConfig& formats, std::uint32_t num_faces, std::uint32_t block_ct_dim)
{
    if constexpr (!tilize_en)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
    }
    else
    {
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, block_ct_dim, FACE_R_DIM, false /* narrow_tile */, num_faces);
    }
}

void unpack_dest_bank_tiles(
    const Operand& buffer_A,
    const FormatConfig& formats,
    std::uint32_t num_tiles,
    std::uint32_t num_faces,
    std::uint32_t block_rt_dim,
    std::uint32_t block_ct_dim)
{
    if constexpr (!tilize_en)
    {
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    else
    {
        for (std::uint32_t row = 0; row < block_rt_dim; ++row)
        {
            const std::uint32_t read_offset = row * block_ct_dim;
            for (std::uint32_t column = 0; column < block_ct_dim; ++column)
            {
                _llk_unpack_tilize_wrapper_(
                    L1_ADDRESS(buffer_A[read_offset]),
                    column,
                    formats.unpack_A_src,
                    formats.unpack_A_dst,
                    block_ct_dim,
                    FACE_R_DIM,
                    num_faces,
                    false /* narrow_tile */);
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
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t block_rt_dim       = params.BLOCK_RT_DIM;
    const std::uint32_t block_ct_dim       = params.BLOCK_CT_DIM;
    const Operand& buffer_A                = params.buffer_A;
#else
    constexpr std::uint32_t block_rt_dim = BLOCK_RT_DIM;
    constexpr std::uint32_t block_ct_dim = BLOCK_CT_DIM;
#endif
    const std::uint32_t num_tiles           = NUM_TILES_IN_BLOCK * NUM_BLOCKS;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * _perf_src_handshake_iters_(tilize_en, num_tiles, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        unpack_dest_bank_init(formats, num_faces, block_ct_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Stand in for the dvalids unpack posts: Blackhole tilize posts SrcB per tile,
            // unpack_A posts SrcB only with FP32 dest. Unpack-to-dest posts none.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_stream_set_valid<true /* set_a */, tilize_en || is_fp32_dest_acc_en /* set_b */, PERF_RUN_TYPE>(src_handshake_iters);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                unpack_dest_bank_tiles(buffer_A, formats, num_tiles, num_faces, block_rt_dim, block_ct_dim);
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (tilize_en)
    {
        _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, num_faces);
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
#include "llk_lib_unpack_wrappers.h"

using namespace ckernel;

void math_dest_bank_block(const FormatConfig& formats, int dst_index, std::uint32_t num_tiles_in_block, const ckernel::TensorShape& tensor_shape)
{
    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
    {
        LLK_ASSERT((dst_index + tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "tile exceeds max dest tiles");
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst_index + tile, formats.math, formats.math, tensor_shape);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
#endif
    const std::uint32_t num_tiles           = NUM_TILES_IN_BLOCK * NUM_BLOCKS;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * _perf_src_handshake_iters_(tilize_en, num_tiles, num_faces);
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            is_int_fpu_en,
            llk_test_pack_mode_v<false /* untilize */, tilize_en>>(tensor_shape, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // Unpack-to-dest has no source handshake, so math keeps its own copies and only drops the Dest sync.
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block = 0; block < NUM_BLOCKS; ++block)
                    {
                        math_dest_bank_block(formats, DST_INDEX, NUM_TILES_IN_BLOCK, tensor_shape);
                    }
                }
            }
            else
            {
                _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(src_handshake_iters);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Unpack-to-dest fills Dest in unpack; math has no source to consume.
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block = 0; block < NUM_BLOCKS; ++block)
                    {
                        math_dest_bank_block(formats, DST_INDEX, NUM_TILES_IN_BLOCK, tensor_shape);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    math_dest_bank_block(formats, DST_INDEX, NUM_TILES_IN_BLOCK, tensor_shape);
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

void pack_dest_bank_block(const Operand& buffer_Res, int dst_index, int block, std::uint32_t num_tiles_in_block)
{
    if constexpr (tilize_en)
    {
        // The Blackhole tilize MOP handles one tile per invocation regardless of its init-time num_tiles.
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                (dst_index + tile < ckernel::get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>()),
                "tile exceeds max dest tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                dst_index + tile, L1_ADDRESS(buffer_Res[block * num_tiles_in_block + tile]));
        }
    }
    else
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(dst_index, L1_ADDRESS(buffer_Res[block * num_tiles_in_block]));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
    const int L1_ACC                       = params.L1_ACC;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
#ifndef ARCH_BLACKHOLE
    (void)L1_ACC;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false /* untilize */, tilize_en>>(
            formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false /* untilize */, tilize_en>, false /* zero_output */>(
            formats.pack_src,
            formats.pack_dst,
            FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            tilize_en ? 1 : NUM_TILES_IN_BLOCK /* num_tiles */);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        reconfigure_packer_l1_acc(L1_ACC);
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
                    pack_dest_bank_block(buffer_Res, DST_INDEX, block, NUM_TILES_IN_BLOCK);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    pack_dest_bank_block(buffer_Res, DST_INDEX, block, NUM_TILES_IN_BLOCK);
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test: Pack tiny tiles from sparse DEST (Tile32x32 slots) to dense L1
// using _llk_pack_block_contiguous_ with a single _llk_pack_ call.
//
// Math uses standard datacopy (Tile32x32 DEST addressing — sparse).
// Pack uses the new block-contiguous MOP that reads from sparse DEST
// slots via W counter and writes dense L1 via auto-increment.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ---------------------------------------------------------------------------
// TRISC0 — UNPACK
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

inline void unpack_tiles(const Operand& buffer_A, const FormatConfig& formats, std::uint32_t total_tiles)
{
    for (std::uint32_t tile = 0; tile < total_tiles; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
            L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t face_r_dim         = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces          = params.num_faces;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const Operand& buffer_A                = params.buffer_A;
#else
    constexpr std::uint32_t face_r_dim         = TEST_FACE_R_DIM;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
#endif
    const std::uint32_t total_tiles         = num_tiles_in_block * num_blocks;
    const std::uint32_t src_handshake_iters = loop_factor * _perf_src_handshake_iters_(false /* tilize_en */, total_tiles, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, num_faces, num_faces);

        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(face_r_dim, num_faces),
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
            _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(src_handshake_iters);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                unpack_tiles(buffer_A, formats, total_tiles);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

// ---------------------------------------------------------------------------
// TRISC1 — MATH (standard datacopy — sparse Tile32x32 DEST slots)
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "params.h"

using namespace ckernel;

inline void math_block(const FormatConfig& formats, const ckernel::TensorShape& tensor_shape, std::uint32_t num_tiles_in_block)
{
    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
            tile, formats.math, formats.math, tensor_shape);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t face_r_dim         = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces          = params.num_faces;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
#else
    constexpr std::uint32_t face_r_dim         = TEST_FACE_R_DIM;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
#endif
    const std::uint32_t total_tiles         = num_tiles_in_block * num_blocks;
    const std::uint32_t src_handshake_iters = loop_factor * _perf_src_handshake_iters_(false /* tilize_en */, total_tiles, num_faces);
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(face_r_dim, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(tensor_shape, formats.math);
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
            _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(src_handshake_iters);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    math_block(formats, tensor_shape, num_tiles_in_block);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// ---------------------------------------------------------------------------
// TRISC2 — PACK (block-contiguous: sparse DEST -> dense L1)
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_pack_block.h"
#endif
#include "params.h"

inline void pack_block(const Operand& buffer_Res, std::uint32_t block, std::uint32_t num_tiles_in_block)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_block_contiguous_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(buffer_Res[block * num_tiles_in_block]), num_tiles_in_block);
#else
    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
    {
        const std::uint32_t res_idx = block * num_tiles_in_block + tile;
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[res_idx]));
    }
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t face_r_dim         = params.TEST_FACE_R_DIM;
    const std::uint32_t tile_c_dim         = params.in0_tile_c_dim;
    const std::uint32_t num_faces          = params.num_faces;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const int l1_acc                       = params.L1_ACC;
    const Operand& buffer_Res              = params.buffer_Res;
#else
    constexpr std::uint32_t face_r_dim         = TEST_FACE_R_DIM;
    constexpr std::uint32_t tile_c_dim         = in0_tile_c_dim;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr int l1_acc                       = L1_ACC;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4, face_r_dim, tile_c_dim, num_faces);

        _llk_pack_init_with_src_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_src, formats.pack_dst, face_r_dim, tile_c_dim, num_faces, false /* partial_face */, false /* narrow_tile */, 1 /* num_tiles */);

        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#ifdef ARCH_BLACKHOLE
        reconfigure_packer_l1_acc(l1_acc);

        // The execute call supplies the tile count at runtime.
        _llk_pack_block_contiguous_mop_config_<>(face_r_dim, num_faces);
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    pack_block(buffer_Res, block, num_tiles_in_block);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

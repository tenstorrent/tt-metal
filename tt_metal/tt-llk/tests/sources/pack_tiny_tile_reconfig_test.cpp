// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test: Full re-init from 32x32 to tiny tiles, then multi-tile block pack.
//
// Flow:
//   1. _llk_pack_hw_configure_ + _llk_pack_init_ for 32x32
//   2. _llk_pack_hw_configure_ + _llk_pack_init_ for the target tiny tile dims
//   3. _llk_pack_block_contiguous_mop_config_ to set up REPLAY + MOP
//   4. _llk_pack_block_contiguous_ — packs all tiles in one call

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
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            params.TEST_FACE_R_DIM,
            params.TEST_FACE_R_DIM,
            params.num_faces,
            params.num_faces);

        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, params.num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    const int total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true /* src A */, is_fp32_dest_acc_en>(LOOP_FACTOR * total_tiles * params.num_faces);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int i = 0; i < total_tiles; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                        L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

// ---------------------------------------------------------------------------
// TRISC1 — MATH
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
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
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, params.num_faces);
    const int num_tiles_in_block            = params.NUM_TILES_IN_BLOCK;
    const int num_blocks                    = params.NUM_BLOCKS;
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true /* src A */, true /* src B */>(LOOP_FACTOR * num_blocks * num_tiles_in_block * params.num_faces);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    for (int tile = 0; tile < num_tiles_in_block; tile++)
                    {
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            false /* unpack_to_dest */>(tile, formats.math, formats.math, tensor_shape);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

                    for (int tile = 0; tile < num_tiles_in_block; tile++)
                    {
                        // Standard datacopy: sparse Tile32x32 DEST slots.
                        // The block-contiguous pack handles the sparse->dense conversion.
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            false /* unpack_to_dest */>(tile, formats.math, formats.math, tensor_shape);
                    }

                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// ---------------------------------------------------------------------------
// TRISC2 — PACK (init 32x32 -> reconfig to tiny -> multi-tile MOP pack)
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_pack_block.h"
#endif
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const int num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const int num_blocks         = params.NUM_BLOCKS;

    {
        START_PERF_MEASURE("INIT")
#ifdef ARCH_BLACKHOLE
    // --- Phase 1: Init for standard 32x32 tiles ---
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);

    _llk_pack_init_with_src_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */, false /* partial_face */, false /* narrow_tile */, 1 /* num_tiles */);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    reconfigure_packer_l1_acc(params.L1_ACC);

    // --- Phase 2: Full re-init for the actual tiny tile dims ---
    // This overwrites the 32x32 config established in Phase 1, testing
    // that the transition from one tile shape to another is correct.
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, params.TEST_FACE_R_DIM, params.in0_tile_c_dim, params.num_faces);

    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, false /* skip_packer_strides */>(
        formats.pack_src, params.TEST_FACE_R_DIM, params.in0_tile_c_dim, params.num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

    // Replace MOP with the block-contiguous version (REPLAY + W-per-tile).
    _llk_pack_block_contiguous_mop_config_<>(params.TEST_FACE_R_DIM, params.num_faces);
#else
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
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
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    if constexpr (sync_with_math)
                    {
                        _llk_packer_wait_for_math_done_();
                    }

#ifdef ARCH_BLACKHOLE
                    _llk_pack_block_contiguous_<DstSync::SyncHalf, is_fp32_dest_acc_en>(
                        0, L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block]), num_tiles_in_block);
#else
                    for (int tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        int res_idx = block * num_tiles_in_block + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[res_idx]));
                    }
#endif

                    if constexpr (sync_with_math)
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

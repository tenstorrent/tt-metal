// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr int MAX_TILES_DEST                    = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        ZONE_SCOPED("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            params->num_faces,
            params->num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, is_fp32_dest_acc_en, reuse_dest_type, unpack_to_dest>(
            params->UNPACK_TRANSPOSE_FACES, params->UNPACK_TRANSPOSE_WITHIN_FACE, FACE_R_DIM, params->num_faces, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // In case of math isolate, we don't want any software synchronization from unpack to math.
            // So we just set/clear valid bits here - which is unavoidable hardware synchronization.
            // When unpack_to_dest is used, we assume the data is immediately ready in destination register.
            // Otherwise, we assume the data is immediately ready in source A/B registers.
            if (!unpack_to_dest)
            {
                // Set valid for source A always.
                // Set valid for source B only if dest_acc is enabled.
                // Works only when unpacking to dest is not used.
                _perf_unpack_loop_set_valid<
                    /* src A */ true,
                    /* src B */ is_fp32_dest_acc_en>(
                    /* iterations*/ params->num_faces * params->TILE_CNT * params->LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int i = 0; i < params->TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, is_fp32_dest_acc_en, reuse_dest_type, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        ZONE_SCOPED("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(params->num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        _llk_math_eltwise_unary_sfpu_init_<SFPU_UNARY_OPERATION>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int i = 0; i < params->TILE_CNT; ++i)
                {
                    // For unpack isolate scenario, math should only perform necessary synchronization and nothing else.
                    if constexpr (unpack_to_dest)
                    {
                        // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                        // "consumed" and can be overwritten with new data.
                        // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                        // this method will perform synchronization only and no actual data copy.
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            i % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                        _perf_math_loop_clear_valid<
                            /* src A */ true,
                            /* src B */ true>(
                            /* iterations*/ params->num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                            // "consumed" and can be overwritten with new data.
                            // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                            // this method will perform synchronization only and no actual data copy.
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                            _perf_math_loop_clear_valid<
                                /* src A */ true,
                                /* src B */ true>(
                                /* iterations*/ params->num_faces);
                        }
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        // When data is not unpacked to dest, math needs to copy data from srcA to dest before starting SFPU operation.
                        // Otherwise, data is immediately ready in destination register.
                        if constexpr (!unpack_to_dest)
                        {
                            LLK_ASSERT(
                                (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }

                        _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(/* dst_index */ block_tile);
                        test_utils::call_sfpu_operation<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(
                            SFPU_UNARY_OPERATION, formats.math);
                        _llk_math_eltwise_unary_sfpu_done_();
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    // Copy from srcA to dest
                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");

                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);

                        // Start SFPU operation
                        _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(/* dst_index */ block_tile);
                        test_utils::call_sfpu_operation<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(
                            SFPU_UNARY_OPERATION, formats.math);
                        _llk_math_eltwise_unary_sfpu_done_();
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

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

        // Configure packer hardware
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * params->num_faces);

#ifdef ARCH_BLACKHOLE
        _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params->num_faces);
#else
        _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, params->num_faces);
#endif
        // Initialize destination for packing
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, /* untilize */ false>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (int loop = 0; loop < params->LOOP_FACTOR; ++loop)
            {
                for (int block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, /* untilize */ false>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

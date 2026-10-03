// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf variant of sources/unpack_A_test.cpp, narrowed to the broadcast unary datacopy.
//
// It exists to measure the math-thread cost of the 32-bit unpack-to-dest broadcast path in
// _llk_math_eltwise_unary_datacopy_ (the `unpack_to_dest && is_32bit_input(...)` branch, which
// issues an explicit MOVD2B/MOVB2D stream instead of the preconfigured MOP). No other perf source
// reaches it: they all pin BROADCAST_TYPE to NONE.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                   = 0;
std::uint32_t pack_sync_tile_dst_ptr            = 0;
std::uint32_t math_sync_tile_dst_index          = 0;
static constexpr std::uint32_t MAX_TILES_DEST   = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

// Only these two run types are implemented. MATH_ISOLATE is the metric this source exists for;
// L1_TO_L1 gives the end-to-end reading. UNPACK_ISOLATE / PACK_ISOLATE / L1_CONGESTION would need
// the math thread to stand in for the unpacker's dest handshake (see the unpack TILE_LOOP note),
// so they are rejected at compile time rather than measured wrong.
static_assert(
    PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_TO_L1,
    "unpack_a_bcast_datacopy_perf supports only PerfRunType::MATH_ISOLATE and PerfRunType::L1_TO_L1");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;

    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;

    const auto& buffer_A = params.buffer_A;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        // acc_to_dest must be false to allow unpack_to_dest (the static assert in
        // llk_unpack_A forbids both together) — matches the functional kernel.
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

        // The producer runs for both supported run types, MATH_ISOLATE included. With
        // unpack_to_dest the datacopy blocks on the UNPACK_TO_DEST / MATH_DONE semaphore pair
        // (math_unpack_to_dest_math_ready / math_unpack_to_dest_tile_ready), which is software
        // sync that _perf_unpack_loop_set_valid's bare SETDVALID cannot satisfy — dropping the
        // producer would hang math instead of isolating it. So MATH_ISOLATE here means "math plus
        // the unpack-to-dest handshake, no packer"; the handshake is identical across compared
        // trees, so a math-stream change still shows up as the delta.
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest (see init) */, reuse_dest_type, unpack_to_dest>(
                    L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif

    // Use B2D for all broadcasts except NONE (data in srcB), A2D for NONE (data in srcA).
    // Mirrors sources/unpack_A_test.cpp so this source instantiates the same template as the
    // correctness sweep in test_bcast.py.
    constexpr DataCopyType copy_type   = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;
    constexpr bool is_int_fpu_en_local = false;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en_local, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    // No _llk_math_wait_for_dest_available_ / _llk_math_dest_section_done_ here: the
                    // packer is idle in this run type, so the MATH_PACK handshake is left out and
                    // the same dest half is rewritten every block.
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math, num_faces);
                    }
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math, num_faces);
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

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const auto& buffer_Res          = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        // MATH_ISOLATE leaves the packer idle so it contributes no dest contention.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK

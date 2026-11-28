// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

static constexpr std::uint32_t MAX_TILES_DEST = ckernel::get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            num_faces_A,
            num_faces_B,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_matmul_init_<>(UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM, FACE_R_DIM, FACE_R_DIM, num_faces_A, num_faces_B, false, false);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    // Golden (LOOP_FACTOR==1) reads unique stimuli tiles. Perf
                    // (LOOP_FACTOR>1) programs the PERF_ADDRESS ring base with tile
                    // index 0. That only removes the caller-side offset;
                    // _llk_unpack_AB_matmul_ still MOP-strides the THCON base (A by
                    // kt per extra output row, B by 1 per extra output column). The
                    // sweep keeps that walk inside PERF_RING_TILES.
                    const bool perf_ring       = LOOP_FACTOR > 1;
                    const std::uint32_t addr_a = perf_ring ? PERF_ADDRESS(PERF_INPUT_A, 0) : L1_ADDRESS(buffer_A[0]);
                    const std::uint32_t addr_b = perf_ring ? PERF_ADDRESS(PERF_INPUT_B, 0) : L1_ADDRESS(buffer_B[0]);
                    const std::uint32_t tile_a = perf_ring ? 0 : j;
                    const std::uint32_t tile_b = perf_ring ? 0 : j * CT_DIM;
                    const std::uint32_t last_a = tile_a + (RT_DIM - 1) * KT_DIM;
                    const std::uint32_t last_b = tile_b + CT_DIM - 1;
                    if (perf_ring)
                    {
                        LLK_ASSERT(last_a < PERF_RING_TILES && last_b < PERF_RING_TILES, "unpack MOP walk exceeds the 16-tile PERF_ADDRESS ring");
                    }
                    else
                    {
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_A[last_a])), "unpack A real-buffer top address is outside L1");
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_B[last_b])), "unpack B real-buffer top address is outside L1");
                    }
                    _llk_unpack_AB_matmul_<>(
                        addr_a,
                        addr_b,
                        tile_a,
                        tile_b,
                        TILE_SIZE_UNPACK_A,
                        TILE_SIZE_UNPACK_B,
                        /* partial face */ false,
                        /* partial face */ false,
                        CT_DIM,
                        RT_DIM,
                        KT_DIM);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            /* tile A */ TILE_R_DIM,
            /* tile A */ TILE_C_DIM,
            /* tile B */ TILE_R_DIM,
            /* tile B */ TILE_C_DIM,
            /* partial face */ false,
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(/* dest_index */ 0, CT_DIM, RT_DIM);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                _llk_math_wait_for_dest_available_<dest_sync>();
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(/* dest_index */ 0, CT_DIM, RT_DIM);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
// after params.h: the shim reads the test's formats and PACK_API_SHIM_* settings
#include "pack_api_shim.h"
#include "pack_api_sweep.h"

// llk_pack_block's forms behind matmul math through the compute API's code (pack_api_shim.h, pack_api_sweep.h), steady state:
// PACK_API's arm into a CB over the perf output ring of pack_sweep_cbd blocks (0: as many as fit), the pack thread marked as
// matmul_block_init marks it, and with PACK_API_SHIM_SFPU as a pack-thread SFPU init marks it behind the matmul.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
#endif
    constexpr std::uint32_t out     = PACK_API_SHIM_OUT;
    constexpr std::uint32_t n       = pack_api_tiles;
    const std::uint32_t ring_blocks = PERF_RING_TILES * 4096 / (n * TILE_SIZE_PACK * 16);
    const std::uint32_t fit_blocks  = ring_blocks != 0 ? ring_blocks : 1;
    const std::uint32_t cb_blocks   = pack_sweep_cbd != 0 && pack_sweep_cbd < fit_blocks ? pack_sweep_cbd : fit_blocks;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
        pack_api_shim::cb_bind(out, PERF_ADDRESS(PERF_OUTPUT, 0) + 1, TILE_SIZE_PACK, cb_blocks * n);
        pack_api_sweep::init<pack_api_arm>(out);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        pack_api_sweep::mark_matmul();
#if defined(PACK_API_SHIM_SFPU)
        pack_api_sweep::mark_sfpu();
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        pack_api_sweep::run<n, is_fp32_dest_acc_en, pack_api_kt, 0, pack_api_arm, pack_api_arm, out, out>(LOOP_FACTOR, 1, 0, [] {}, [] {});
        PROFILER_SYNC();
    }
}

#endif

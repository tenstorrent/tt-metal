// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

#include "llk_unpack_A.h"
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
        // A tiles 0..KT_DIM-2 seed the exp spare Dest tiles.
        _llk_unpack_A_init_<>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces_A),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        for (std::uint32_t t = 0; t + 1 < KT_DIM; t++)
        {
            _llk_unpack_A_<>(L1_ADDRESS(buffer_A[t]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        _llk_unpack_A_uninit_();
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
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_matmul.h"
#include "sfpu_operations.h"

// Accurate EXP_OP on a bf16 Dest records its body into the replay buffer; it runs between two K
// steps of one matmul, as a fused activation runs between matmul_block calls. K step j > 0 runs it
// on spare Dest tile EXP_SPARE_DEST_TILE - (j - 1), seeded with A tile j - 1.
constexpr std::uint32_t EXP_SPARE_DEST_TILE = MAX_TILES_DEST - 1;

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
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        LLK_ASSERT(LOOP_FACTOR == 1, "one Dest section: the spare tiles are seeded once");
        LLK_ASSERT(get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) + KT_DIM - 1 < MAX_TILES_DEST, "exp spare tiles overlap the matmul output tiles");
        _llk_math_wait_for_dest_available_<dest_sync>();
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4 /* num_faces */), formats.math);
        for (std::uint32_t t = 0; t + 1 < KT_DIM; t++)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en>(EXP_SPARE_DEST_TILE - t, formats.math, formats.math);
        }
        _llk_math_eltwise_unary_datacopy_uninit_();
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            /* tile A */ TILE_R_DIM,
            /* tile A */ TILE_C_DIM,
            /* tile B */ TILE_R_DIM,
            /* tile B */ TILE_C_DIM,
            /* partial face */ false,
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM);
        test_utils::call_unary_sfpu_operation_init<
            EXP_OP,
            false /* APPROX_MODE */,
            is_fp32_dest_acc_en,
            32 /* ITERATIONS */,
            false /* FAST_MODE */,
            false /* STABLE_SORT */,
            EXP_CLAMP_NEGATIVE>(formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "functional test only");
        for (std::uint32_t j = 0; j < KT_DIM; j++)
        {
            if (j > 0)
            {
                test_utils::call_unary_sfpu_operation<
                    dest_sync,
                    is_fp32_dest_acc_en,
                    EXP_OP,
                    false /* APPROX_MODE */,
                    is_fp32_dest_acc_en,
                    32 /* ITERATIONS */,
                    false /* FAST_MODE */,
                    false /* STABLE_SORT */,
                    EXP_CLAMP_NEGATIVE>(EXP_SPARE_DEST_TILE - (j - 1), formats.math);
            }
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(/* dest_index */ 0, CT_DIM, RT_DIM);
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t CT_DIM         = params.CT_DIM;
    const std::uint32_t RT_DIM         = params.RT_DIM;
    const std::uint32_t KT_DIM         = params.KT_DIM;
    const Operand& buffer_Res          = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    const std::uint32_t tile_index = tile % MAX_TILES_DEST;
                    LLK_ASSERT(
                        (tile_index < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles for matmul");
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
                }
                // Wormhole: release the dest section as production's tile_regs_release does, so every section starts with drained packers
                if constexpr (PERF_PACK_SECTION_RELEASE)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                // Pack dest occupancy (RT×CT).
                for (std::uint32_t i = 0; i < CT_DIM * RT_DIM; i++)
                {
                    LLK_ASSERT((i < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "i exceeds max dest tiles");
                    // Golden packs unique result tiles. Perf K=32 dest-fill Float32
                    // places buffer_Res past L1; use the PERF_ADDRESS ring like matmul_perf.
                    const std::uint32_t addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, i) : L1_ADDRESS(buffer_Res[i]);
                    if (LOOP_FACTOR == 1)
                    {
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[i])), "pack result real-buffer address is outside L1");
                    }
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, addr);
                }
                // Exp spare tiles follow the matmul result.
                for (std::uint32_t t = 0; t + 1 < KT_DIM; t++)
                {
                    const std::uint32_t res_tile = CT_DIM * RT_DIM + t;
                    LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[res_tile])), "pack result real-buffer address is outside L1");
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(MAX_TILES_DEST - 1 - t, L1_ADDRESS(buffer_Res[res_tile]));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

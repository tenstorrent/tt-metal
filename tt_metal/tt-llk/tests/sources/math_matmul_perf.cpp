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
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = (dest_sync == ckernel::DstSync::SyncFull ? 16 : 8) / (is_fp32_dest_acc_en ? 2 : 1);

struct MatmulBlockDimensions
{
    std::uint32_t ct;
    std::uint32_t rt;
};

inline MatmulBlockDimensions select_matmul_block_dimensions(std::uint32_t ct_dim, std::uint32_t rt_dim)
{
    MatmulBlockDimensions best = {1, 1};

    for (std::uint32_t rt = 1; rt <= rt_dim; ++rt)
    {
        if (rt_dim % rt != 0)
        {
            continue;
        }

        for (std::uint32_t ct = 1; ct <= ct_dim; ++ct)
        {
            if (ct_dim % ct != 0 || ct * rt > MAX_TILES_DEST)
            {
                continue;
            }

            const std::uint32_t candidate_tiles = ct * rt;
            const std::uint32_t best_tiles      = best.ct * best.rt;
            const std::uint32_t candidate_loads = ct + rt;
            const std::uint32_t best_loads      = best.ct + best.rt;

            if (candidate_tiles > best_tiles || (candidate_tiles == best_tiles && candidate_loads < best_loads) ||
                (candidate_tiles == best_tiles && candidate_loads == best_loads && ct > best.ct))
            {
                best = {ct, rt};
            }
        }
    }

    return best;
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;

    const bool PARTIAL_FACE_A = params.PARTIAL_FACE_A;
    const bool PARTIAL_FACE_B = params.PARTIAL_FACE_B;

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

    const MatmulBlockDimensions block_dimensions = select_matmul_block_dimensions(CT_DIM, RT_DIM);
    const std::uint32_t BLOCK_CT_DIM             = block_dimensions.ct;
    const std::uint32_t BLOCK_RT_DIM             = block_dimensions.rt;

    {
        START_PERF_MEASURE("INIT")
        LLK_ASSERT(CT_DIM % BLOCK_CT_DIM == 0 && RT_DIM % BLOCK_RT_DIM == 0, "Matmul output grid must be evenly divisible into destination blocks");
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B, // In1
            num_faces_A, // In0
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_matmul_init_<>(
            UNPACK_TRANSPOSE_FACES,
            BLOCK_CT_DIM,
            BLOCK_RT_DIM,
            KT_DIM,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B,     // In1
            num_faces_A,     // In0
            PARTIAL_FACE_B,  // In1
            PARTIAL_FACE_A); // In0
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        _perf_unpack_matmul_mock(1, BLOCK_RT_DIM, KT_DIM, BLOCK_CT_DIM);
                    }
                }
            }
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        for (std::uint32_t j = 0; j < KT_DIM; j++)
                        {
                            const std::uint32_t src_a_tile = block_row * KT_DIM + j;
                            const std::uint32_t src_b_tile = j * CT_DIM + block_col;
                            _llk_unpack_AB_matmul_<>(
                                L1_ADDRESS(buffer_A[0]),
                                L1_ADDRESS(buffer_B[0]),
                                src_a_tile,
                                src_b_tile,
                                TILE_SIZE_UNPACK_A,
                                TILE_SIZE_UNPACK_B,
                                PARTIAL_FACE_B, // In1
                                PARTIAL_FACE_A, // In0
                                BLOCK_CT_DIM,
                                BLOCK_RT_DIM,
                                KT_DIM);
                        }
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
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in0_tile_c_dim = params.in0_tile_c_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;
    const std::uint32_t in1_tile_c_dim = params.in1_tile_c_dim;

    const bool PARTIAL_FACE_MATH = params.PARTIAL_FACE_MATH;

    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const int DST_INDEX             = params.DST_INDEX;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    const MatmulBlockDimensions block_dimensions = select_matmul_block_dimensions(CT_DIM, RT_DIM);
    const std::uint32_t BLOCK_CT_DIM             = block_dimensions.ct;
    const std::uint32_t BLOCK_RT_DIM             = block_dimensions.rt;

    {
        START_PERF_MEASURE("INIT")
        LLK_ASSERT(DST_INDEX + BLOCK_CT_DIM * BLOCK_RT_DIM <= MAX_TILES_DEST, "Matmul block exceeds destination capacity");
        LLK_ASSERT(CT_DIM % BLOCK_CT_DIM == 0 && RT_DIM % BLOCK_RT_DIM == 0, "Matmul output grid must be evenly divisible into destination blocks");
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, PARTIAL_FACE_MATH, UNPACK_TRANSPOSE_FACES, BLOCK_CT_DIM, BLOCK_RT_DIM);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        _perf_math_matmul_mock(1, BLOCK_RT_DIM, KT_DIM, BLOCK_CT_DIM);
                    }
                }
            }
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        for (std::uint32_t j = 0; j < KT_DIM; j++)
                        {
                            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, BLOCK_CT_DIM, BLOCK_RT_DIM);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                        for (std::uint32_t j = 0; j < KT_DIM; j++)
                        {
                            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, BLOCK_CT_DIM, BLOCK_RT_DIM);
                        }
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const bool PARTIAL_FACE_PACK       = params.PARTIAL_FACE_PACK;
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const int num_faces                = params.num_faces;
    const int DST_INDEX                = params.DST_INDEX;
    const std::uint32_t CT_DIM         = params.CT_DIM;
    const std::uint32_t RT_DIM         = params.RT_DIM;
#endif

    const MatmulBlockDimensions block_dimensions = select_matmul_block_dimensions(CT_DIM, RT_DIM);
    const std::uint32_t BLOCK_CT_DIM             = block_dimensions.ct;
    const std::uint32_t BLOCK_RT_DIM             = block_dimensions.rt;

    {
        START_PERF_MEASURE("INIT")
        LLK_ASSERT(DST_INDEX + BLOCK_CT_DIM * BLOCK_RT_DIM <= MAX_TILES_DEST, "Matmul block exceeds destination capacity");
        LLK_ASSERT(CT_DIM % BLOCK_CT_DIM == 0 && RT_DIM % BLOCK_RT_DIM == 0, "Matmul output grid must be evenly divisible into destination blocks");
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src,
            formats.pack_dst,
            TILE_SIZE_PACK,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            PARTIAL_FACE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        for (std::uint32_t tile = 0; tile < BLOCK_CT_DIM * BLOCK_RT_DIM; tile++)
                        {
                            const std::uint32_t local_row   = tile / BLOCK_CT_DIM;
                            const std::uint32_t local_col   = tile % BLOCK_CT_DIM;
                            const std::uint32_t output_tile = (block_row + local_row) * CT_DIM + block_col + local_col;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, output_tile));
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_row = 0; block_row < RT_DIM; block_row += BLOCK_RT_DIM)
                {
                    for (std::uint32_t block_col = 0; block_col < CT_DIM; block_col += BLOCK_CT_DIM)
                    {
                        _llk_packer_wait_for_math_done_();
                        for (std::uint32_t tile = 0; tile < BLOCK_CT_DIM * BLOCK_RT_DIM; tile++)
                        {
                            const std::uint32_t local_row   = tile / BLOCK_CT_DIM;
                            const std::uint32_t local_col   = tile % BLOCK_CT_DIM;
                            const std::uint32_t output_tile = (block_row + local_row) * CT_DIM + block_col + local_col;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, output_tile));
                        }
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

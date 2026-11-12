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
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
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
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B, // in1
            num_faces_A, // in0
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_matmul_init_<>(
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM,
            KT_DIM,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B,    // in1
            num_faces_A,    // in0
            PARTIAL_FACE_B, // in1
            PARTIAL_FACE_A);
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
            _perf_unpack_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_unpack_AB_matmul_<>(
                            L1_ADDRESS(buffer_A[0]),
                            L1_ADDRESS(buffer_B[0]),
                            j,
                            j * CT_DIM,
                            TILE_SIZE_UNPACK_B,
                            TILE_SIZE_UNPACK_A,
                            PARTIAL_FACE_B,
                            PARTIAL_FACE_A,
                            CT_DIM,
                            RT_DIM,
                            KT_DIM);
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

    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int DST_INDEX                    = (params.DST_INDEX & 0xff);
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, PARTIAL_FACE_MATH, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    LLK_ASSERT(
                        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Matmul block exceeds destination capacity");
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
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
#include "pack_sweep_helpers.h"

#ifndef MATMUL_PACK_FORM
#define MATMUL_PACK_FORM 0
#endif
constexpr bool sw_rmw = dest_sync == DstSync::SyncFull;

// MATMUL_PACK_FORM 0 per-tile packs, 2 the closed block that programs the output, 3 the closed block with the W offset (output programmed once in INIT).
inline void matmul_pack_out(
    std::uint32_t form, std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces)
{
    if (form == 2)
    {
        _llk_pack_block_closed_<dest_sync, is_fp32_dest_acc_en, sw_rmw>(tile_index, address, num_tiles, tile_stride, num_faces);
    }
    else
    {
        _llk_pack_block_closed_w_mop_<dest_sync, is_fp32_dest_acc_en>(tile_index, 0, num_tiles, num_faces);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim  = params.in0_tile_r_dim;
    const bool PARTIAL_FACE_PACK        = params.PARTIAL_FACE_PACK;
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK  = params.TILE_SIZE_PACK;
    const int num_faces                 = params.num_faces;
    const int DST_INDEX                 = (params.DST_INDEX & 0xff);
    const volatile std::uint32_t form_v = (static_cast<std::uint32_t>(params.DST_INDEX) >> 8) & 3;
    const std::uint32_t form_rt         = form_v;
    // pk-limit5: DST_INDEX bit 10 with form 3 = the W offset under cb0ac's rule, the output a CB of two blocks: a W block continues only right behind a
    // W block of the output at the next address, otherwise it programs the base (and the W stride) first.
    const volatile std::uint32_t wcont_v = (static_cast<std::uint32_t>(params.DST_INDEX) >> 10) & 1;
    const bool wcont                     = wcont_v != 0;
    std::uint32_t w_next = 0, w_tiles = 0;
    bool w_last                            = false;
    const std::uint32_t CT_DIM             = params.CT_DIM;
    const std::uint32_t RT_DIM             = params.RT_DIM;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
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
        if constexpr (sw_rmw)
        {
            _llk_pack_rmw_dest_init_<true>();
        }
        if (form_rt == 3 && !wcont)
        {
            _llk_pack_block_w_init_(L1_ADDRESS(buffer_Res[0]), (L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0])) << 4);
        }
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
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                    {
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
#if defined(ARCH_BLACKHOLE)
                    if (form_rt == 1)
                    {
                        // pk-limit5: main's per-tile pack (no byte-wise RMW)
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default, false, false>(
                                DST_INDEX + tile, L1_ADDRESS(buffer_Res[result_tile]));
                        }
                    }
                    else if (form_rt == 3 && wcont)
                    {
                        const std::uint32_t stride = L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0]);
                        const std::uint32_t addr   = L1_ADDRESS(buffer_Res[(loop & 1) * NUM_TILES_IN_BLOCK]);
                        if (!(w_last && w_next == addr && w_tiles + NUM_TILES_IN_BLOCK <= 127))
                        {
                            pack_sweep_w_reprogram(addr, stride << 4);
                            w_tiles = 0;
                            w_next  = addr;
                        }
                        const std::uint32_t out_tile = w_tiles;
                        w_tiles += NUM_TILES_IN_BLOCK;
                        w_next += NUM_TILES_IN_BLOCK * stride;
                        w_last = true;
                        _llk_pack_block_closed_w_mop_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX, out_tile, NUM_TILES_IN_BLOCK, num_faces);
                    }
                    else if (form_rt != 0)
                    {
                        matmul_pack_out(
                            form_rt,
                            DST_INDEX,
                            L1_ADDRESS(buffer_Res[block * NUM_TILES_IN_BLOCK]),
                            NUM_TILES_IN_BLOCK,
                            (L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0])) << 4,
                            num_faces);
                    }
                    else
#endif
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                            _llk_pack_<
                                dest_sync,
                                is_fp32_dest_acc_en,
                                ckernel::PackMode::Default,
                                false,
                                sw_rmw && !(!is_fp32_dest_acc_en && sweep_float32_out)>(DST_INDEX + tile, L1_ADDRESS(buffer_Res[result_tile]));
                        }
                    }
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

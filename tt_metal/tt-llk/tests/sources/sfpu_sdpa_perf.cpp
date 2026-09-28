// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf driver for the SDPA column-vector SFPU bodies in metal's experimental llk_sfpu
// (ckernel_sfpu_sdpa.h, ckernel_sfpu_sdpa_fw.h), selected by SDPA_PERF_OP (SdpaPerfOp in
// helpers/llk_params.py). The functional tests are test_sfpu_sdpa.py / test_sfpu_sdpa_fw.py;
// this file mirrors their init and dispatch (VectorMode::C, one call per tile) inside the
// eltwise_unary_sfpu_perf.cpp loop shape: one A2D datacopy per tile, then the SFPU body, so
// `mean(MATH_ISOLATE)` of TILE_LOOP is cycles per datacopy + one body call.
//
// The correction body works on five DEST tiles from its base (four in the FP32 half-sync layout);
// it is always dispatched at tile 0, which the block's datacopies keep populated. Its cost does not
// depend on the values, so which tile holds what does not matter here.

#include <algorithm>
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

static constexpr std::uint32_t MAX_TILES_DEST   = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t NUM_FACES        = 4;

// Mirrors SdpaPerfOp in helpers/llk_params.py.
constexpr int PERF_EXP_ACCURATE = 0;
constexpr int PERF_EXP_POLY     = 1;
constexpr int PERF_CORRECTION   = 2;
constexpr int PERF_FW_EXP       = 3;
static_assert(SDPA_PERF_OP >= PERF_EXP_ACCURATE && SDPA_PERF_OP <= PERF_FW_EXP, "unhandled SDPA_PERF_OP");

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
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_A_src, formats.unpack_A_dst, formats.unpack_A_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
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
            // Math does one A2D datacopy per face; hand it a valid srcA bank per face. With a 32-bit Dest the
            // datacopy also waits on srcB, as in eltwise_unary_sfpu_perf.cpp.
            _perf_unpack_loop_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en>(NUM_FACES * TILE_CNT * LOOP_FACTOR);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, i), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// ckernel_sfpu_sdpa.h needs these.
static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;
static constexpr bool APPROX         = APPROX_MODE;
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"

using namespace ckernel;

// FP32 half-sync has four DEST tiles, so the correction body reuses its cur_max output tile for
// worker_sum there, exactly as ttnn's correction_block and test_sfpu_sdpa.py select it.
constexpr bool REUSE_CUR_MAX_TILE = is_fp32_dest_acc_en && DST_SYNC_MODE == DstSync::SyncHalf;

// Same inits as sources/sfpu_sdpa_test.cpp and sources/sfpu_sdpa_fw_test.cpp.
inline void sdpa_perf_op_init()
{
    if constexpr (SDPA_PERF_OP == PERF_FW_EXP)
    {
        _llk_math_eltwise_unary_sfpu_init_once_();
    }
    else
    {
        sfpu::exp_init<
            SDPA_PERF_OP != PERF_EXP_POLY /* APPROXIMATION_MODE */,
            0x3F800000 /* scale, unused here */,
            true /* CLAMP_NEGATIVE */,
            is_fp32_dest_acc_en>();
    }
}

inline void sdpa_perf_op(const std::uint32_t dst_index)
{
    if constexpr (SDPA_PERF_OP == PERF_EXP_ACCURATE)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_exponential_first_column<true /* SDPA_EXP_APPROX_MODE */, EXP_SCALE_BF16, is_fp32_dest_acc_en>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_PERF_OP == PERF_EXP_POLY)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_exponential_first_column<false /* SDPA_EXP_APPROX_MODE */, EXP_SCALE_BF16, is_fp32_dest_acc_en>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_PERF_OP == PERF_CORRECTION)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_fused_max_sub_exp_add_tile<is_fp32_dest_acc_en, REUSE_CUR_MAX_TILE>, 0, VectorMode::C, static_cast<int>(EXP_SCALE_BF16));
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_exponential_first_column<EXP_SCALE_BF16, is_fp32_dest_acc_en>, dst_index, VectorMode::C);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(NUM_FACES, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        sdpa_perf_op_init();
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
            _perf_math_loop_clear_valid<true, false>(NUM_FACES * TILE_CNT * LOOP_FACTOR);
            return;
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
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                        sdpa_perf_op(block_tile);
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
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                        sdpa_perf_op(block_tile);
                    }
                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
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
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

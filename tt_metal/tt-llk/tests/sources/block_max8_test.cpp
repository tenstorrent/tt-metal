// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK
#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"

/**
 * @brief Feed full BF16 tiles to SrcA using the ordinary unpacker.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<false>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            false /* transpose_of_faces */, false /* within_face_16x16_transpose */, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_stream_set_valid<true /* set_a */, false /* set_b */, PERF_RUN_TYPE>(
                LOOP_FACTOR * _perf_src_handshake_iters_(false /* tilize_en */, TILE_CNT, TILE_NUM_FACES));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_block_max8.h"
#include "sfpu/experimental/ckernel_sfpu_fill_negative_zero.h"

/**
 * @brief Encode exact negative zero as -1 so the packer cannot canonicalize it to +0.
 */
inline void encode_negative_zero()
{
    for (std::uint32_t row = 0; row < 8; row += 4)
    {
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            sfpi::vFloat value = sfpi::dst_reg[row / 2 + parity];
            v_if (sfpi::as<sfpi::vUInt>(value) == 0x80000000u)
            {
                value = -1.0f;
            }
            v_endif;
            sfpi::dst_reg[row / 2 + parity] = value;
        }
    }
}

/**
 * @brief Fill eight DST slots, then run block_max8 in place on one slot; the others detect stray writes.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR          = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT             = params.TILE_CNT;
    const std::uint32_t DST_INDEX            = params.DST_INDEX;
    const std::uint32_t VALID_SCORES         = params.VALID_SCORES;
    const std::uint32_t INJECT_NEGATIVE_ZERO = params.INJECT_NEGATIVE_ZERO;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<false>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, false, BroadcastType::NONE, false, PackMode::Default>(
            ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_stream_clear_valid<true /* clear_a */, false /* clear_b */, PERF_RUN_TYPE>(
                LOOP_FACTOR * _perf_src_handshake_iters_(false /* tilize_en */, TILE_CNT, TILE_NUM_FACES));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t base = 0; base < TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, false, BroadcastType::NONE, false>(
                            slot, formats.math, formats.math);
                    }
                    if (INJECT_NEGATIVE_ZERO)
                    {
                        _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_fill_negative_zero_, DST_INDEX, VectorMode::RC_custom);
                    }
                    _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_block_max8_, DST_INDEX, VectorMode::RC_custom, VALID_SCORES);
                    if (INJECT_NEGATIVE_ZERO)
                    {
                        _llk_math_eltwise_unary_sfpu_params_(encode_negative_zero, DST_INDEX, VectorMode::RC_custom);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
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

/**
 * @brief Pack all eight slots; the pooled result is the first 128 elements of its tile.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<false, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_R_DIM * TILE_C_DIM /* tile_size */);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, false, PackMode::Default>();
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t base = 0; base < TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
                    {
                        _llk_pack_<DstSync::SyncHalf, false, PackMode::Default>(slot, L1_ADDRESS(buffer_Res[base + slot]));
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_PACK_SECTION_RELEASE)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif

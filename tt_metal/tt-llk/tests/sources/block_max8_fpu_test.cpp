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

// Call k of an acquire pools DST slot (DST_INDEX + CALL_STRIDE * k) % DEST_NUM_TILES_FP16_HALF.
constexpr std::uint32_t CALL_STRIDE = 3;

#ifdef LLK_TRISC_UNPACK
#include "experimental/llk_unpack_block_max8_fpu.h"
#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"

/**
 * @brief Feed full BF16 tiles to SrcA, then the FPU block-max sources FPU_CALLS times per batch of DST slots.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t FPU_CALLS   = params.FPU_CALLS;
    const Operand& buffer_A         = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
            false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t base = 0; base < TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
                {
                    _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(
                        _perf_src_handshake_iters_(false /* tilize_en */, DEST_NUM_TILES_FP16_HALF, TILE_NUM_FACES));
                    _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(FPU_CALLS);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                    if (tile % DEST_NUM_TILES_FP16_HALF == DEST_NUM_TILES_FP16_HALF - 1)
                    {
                        for (std::uint32_t call = 0; call < FPU_CALLS; ++call)
                        {
                            _llk_unpack_block_max8_fpu_();
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
#include "experimental/llk_math_block_max8_fpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_fill_negative_zero.h"

/**
 * @brief Make an exponent-zero result (-0.0 or a denormal) that leaks through the FPU path observable through the packer.
 *
 * GMPOOL flushes exponent-zero inputs to +0.0, so the host expects +0.0. The
 * packer also flushes them, so this encodes every nonzero exponent-zero bit
 * pattern as -1 before packing, and a leaked value fails the check instead of
 * passing as zero.
 */
inline void encode_exponent_zero()
{
    for (std::uint32_t row = 0; row < 8; row += 4)
    {
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            sfpi::vFloat value = sfpi::dst_reg[row / 2 + parity];
            v_if ((sfpi::as<sfpi::vUInt>(value) & 0x7F800000u) == 0u && sfpi::as<sfpi::vUInt>(value) != 0u)
            {
                value = -1.0f;
            }
            v_endif;
            sfpi::dst_reg[row / 2 + parity] = value;
        }
    }
}

/**
 * @brief Replace the host markers 0.375 and -0.375 with BF16 denormals in the current DST tile.
 *
 * The unpack/datacopy path flushes denormal stimuli, so the host sends normal
 * markers. 0.375 becomes 0x007F and -0.375 becomes 0x8001.
 */
inline void inject_denormals()
{
    for (std::uint32_t row = 0; row < TILE_NUM_FACES * FACE_R_DIM; row += 4)
    {
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            sfpi::vFloat value = sfpi::dst_reg[row / 2 + parity];
            v_if (value == 0.375f)
            {
                value = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x007F0000u));
            }
            v_elseif (value == -0.375f)
            {
                value = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x80010000u));
            }
            v_endif;
            sfpi::dst_reg[row / 2 + parity] = value;
        }
    }
}

/**
 * @brief Populate eight DST slots, then run the FPU block max in place on FPU_CALLS slots per acquire.
 *
 * Surrounding tiles remain occupied to detect writes outside the selected tiles.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR          = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT             = params.TILE_CNT;
    const std::uint32_t FPU_CALLS            = params.FPU_CALLS;
    const std::uint32_t DST_INDEX            = params.DST_INDEX;
    const std::uint32_t VALID_SCORES         = params.VALID_SCORES;
    const std::uint32_t INJECT_NEGATIVE_ZERO = params.INJECT_NEGATIVE_ZERO;
    const std::uint32_t INJECT_DENORMAL      = params.INJECT_DENORMAL;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
            ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
        _llk_math_block_max8_fpu_init_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t base = 0; base < TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
                {
                    _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(
                        _perf_src_handshake_iters_(false /* tilize_en */, DEST_NUM_TILES_FP16_HALF, TILE_NUM_FACES));
                    _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(FPU_CALLS);
                }
            }
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
                        _llk_math_eltwise_unary_datacopy_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            false /*unpack_to_dest*/>(slot, formats.math, formats.math);
                    }
                    for (std::uint32_t call = 0; call < FPU_CALLS; ++call)
                    {
                        const std::uint32_t dst_index = (DST_INDEX + CALL_STRIDE * call) % DEST_NUM_TILES_FP16_HALF;
                        if (INJECT_NEGATIVE_ZERO)
                        {
                            _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_fill_negative_zero_, dst_index, VectorMode::RC_custom);
                        }
                        if (INJECT_DENORMAL)
                        {
                            _llk_math_eltwise_unary_sfpu_params_(inject_denormals, dst_index, VectorMode::RC_custom);
                        }
                        const std::uint32_t zero_flag = math::src_zero_flag_hw;
                        _llk_math_block_max8_fpu_(dst_index, VALID_SCORES);
                        LLK_ASSERT(math::src_zero_flag_hw == zero_flag, "FPU block max8 changed the Src zero flag");
                        if (INJECT_NEGATIVE_ZERO || INJECT_DENORMAL)
                        {
                            _llk_math_eltwise_unary_sfpu_params_(encode_exponent_zero, dst_index, VectorMode::RC_custom);
                        }
                    }
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
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

/**
 * @brief Pack all occupied slots with the ordinary full-tile packer for host inspection.
 *
 * The pooled result occupies the first 128 physical elements of its tile.
 * Remaining elements of that tile are scratch; other tiles must be unchanged.
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
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_R_DIM * TILE_C_DIM /*tile_size*/);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
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
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(slot, L1_ADDRESS(buffer_Res[base + slot]));
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_PACK_SECTION_RELEASE)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif

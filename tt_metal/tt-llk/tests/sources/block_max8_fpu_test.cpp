// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

using namespace ckernel;
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Call k of an acquire pools DST slot (DST_INDEX + CALL_STRIDE * k) % DEST_NUM_TILES_FP16_HALF.
constexpr std::uint32_t CALL_STRIDE = 3;

#ifdef LLK_TRISC_UNPACK
#include "experimental/llk_unpack_block_max8_fpu.h"
#include "llk_unpack_A.h"

/**
 * @brief Feed full BF16 tiles to SrcA, then the FPU block-max sources FPU_CALLS times per batch of DST slots.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
        false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
            L1_ADDRESS(params.buffer_A[t]), formats.unpack_A_src, formats.unpack_A_dst);
        if (t % DEST_NUM_TILES_FP16_HALF == DEST_NUM_TILES_FP16_HALF - 1)
        {
            for (std::uint32_t call = 0; call < params.FPU_CALLS; ++call)
            {
                _llk_unpack_block_max8_fpu_();
            }
        }
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "experimental/llk_math_block_max8_fpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_fill_negative_zero.h"

/**
 * @brief Make a negative-zero bit pattern that leaks through the FPU path observable through the packer.
 *
 * GMPOOL flushes -0.0 to +0.0, so the host expects +0.0. This encodes an exact
 * -0.0 as -1 before packing, which otherwise canonicalizes zeros, so a leaked
 * -0.0 fails the check instead of passing as zero.
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
 * @brief Populate eight DST slots, then run the FPU block max in place on FPU_CALLS slots per acquire.
 *
 * Surrounding tiles remain occupied to detect writes outside the selected tiles.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_block_max8_fpu_init_();
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false /*unpack_to_dest*/>(
                slot, formats.math, formats.math);
        }
        for (std::uint32_t call = 0; call < params.FPU_CALLS; ++call)
        {
            const std::uint32_t dst_index = (params.DST_INDEX + CALL_STRIDE * call) % DEST_NUM_TILES_FP16_HALF;
            if (params.INJECT_NEGATIVE_ZERO)
            {
                _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_fill_negative_zero_, dst_index, VectorMode::RC_custom);
            }
            _llk_math_block_max8_fpu_(dst_index, params.VALID_SCORES);
            if (params.INJECT_NEGATIVE_ZERO)
            {
                _llk_math_eltwise_unary_sfpu_params_(encode_negative_zero, dst_index, VectorMode::RC_custom);
            }
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_R_DIM * TILE_C_DIM /*tile_size*/);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(slot, L1_ADDRESS(params.buffer_Res[base + slot]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

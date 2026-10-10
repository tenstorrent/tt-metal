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

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"

/**
 * @brief Feed full BF16 tiles to SrcA using the ordinary unpacker.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<false>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_A[t]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_block_max8.h"

/**
 * @brief Inject negative zero directly into DST to isolate SFPU bit preservation.
 *
 * The input unpack/datacopy path may canonicalize zero signs. This test-only
 * prelude supplies exact negative-zero bits to every score before masking and
 * pooling. The unary dispatcher has already selected the target DST tile.
 */
inline void fill_negative_zero()
{
    sfpi::vFloat negative_zero = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x80000000u));
    for (std::uint32_t row = 0; row < TILE_NUM_FACES * FACE_R_DIM; row += 4)
    {
        sfpi::dst_reg[row / 2]     = negative_zero;
        sfpi::dst_reg[row / 2 + 1] = negative_zero;
    }
}

/**
 * @brief Make a preserved negative-zero bit pattern observable through the packer.
 *
 * Encode only exact negative zero as -1 before packing the compacted result.
 * Positive zero stays zero, so it cannot satisfy the host's negative-zero check.
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
 * @brief Populate eight DST slots, then run the SFPU body in place on one runtime-selected slot.
 *
 * The standard unary dispatcher supplies DST addressing and synchronization.
 * Surrounding tiles remain occupied to detect writes outside the selected tile.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<false>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, false, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, false, BroadcastType::NONE, false>(slot, formats.math, formats.math);
        }
        if (params.INJECT_NEGATIVE_ZERO)
        {
            _llk_math_eltwise_unary_sfpu_params_(fill_negative_zero, params.DST_INDEX, VectorMode::RC_custom);
        }
        _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_block_max8_, params.DST_INDEX, VectorMode::RC_custom, params.VALID_SCORES);
        if (params.INJECT_NEGATIVE_ZERO)
        {
            _llk_math_eltwise_unary_sfpu_params_(encode_negative_zero, params.DST_INDEX, VectorMode::RC_custom);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
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
    _llk_pack_hw_configure_<false, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_R_DIM * TILE_C_DIM);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, false, PackMode::Default>();
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_pack_<DstSync::SyncHalf, false, PackMode::Default>(slot, L1_ADDRESS(params.buffer_Res[base + slot]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
}
#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated driver for the Welford SFPU kernel (ckernel_sfpu_welfords.h via
// llk_math_welfords_sfpu.h / llk_math_welfords_sfpu_params.h). It mirrors the
// call sequence of ttnn's welford_reduce_h compute kernel and of the layernorm
// Welford kernels:
//   * each input tile is copied to dst index 0 and folded into the running
//     per-column mean / M2 held in LREG4 / LREG5 (tile row = sample, 32 columns
//     in parallel; consecutive tiles continue the sample sequence),
//   * the last tile optionally goes through the partial-tile path
//     (WELFORD_LAST_START_ROW / WELFORD_LAST_NUM_ROWS),
//   * the state is finalized into dst index 1 (mean) and dst index 2 (variance,
//     or M2 when WELFORD_FINALIZE_RAW), which are the two packed result tiles.
// The llk_math_welfords_sfpu_* entry points in the metal llk_api layer are thin
// SAN_HOOK wrappers around exactly these _llk_ calls.

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

static constexpr std::uint32_t WELFORD_INPUT_DST_INDEX = 0;
static constexpr std::uint32_t WELFORD_MEAN_DST_INDEX  = 1; // variance / M2 lands in WELFORD_MEAN_DST_INDEX + 1

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

using namespace ckernel;

// Reciprocal LUT (entry i = fp32 bits of 1.0f / (i + 1)), filled on the TRISC with the same
// correctly rounded soft-float division the reciprocal_size == 0 fallback uses, so both
// variants see identical reciprocals.
static std::array<std::uint32_t, WELFORD_LUT_SIZE> welford_reciprocal_lut;

static void fill_reciprocal_lut()
{
    if constexpr (WELFORD_LUT_SIZE > 0)
    {
        for (std::uint32_t i = 0; i < WELFORD_LUT_SIZE; ++i)
        {
            const float reciprocal = 1.0f / static_cast<float>(i + 1);
            std::uint32_t bits;
            __builtin_memcpy(&bits, &reciprocal, sizeof(bits));
            welford_reciprocal_lut[i] = bits;
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    fill_reciprocal_lut();

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // The finalize helpers only write rows 0-3 of faces 0/1 of the two result tiles; clear Dest so
    // the rest of the packed result is deterministic (the result buffers are hashed for A/B checks).
    TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);

    // Same order as ttnn welford_reduce_h: welford_init() (replay record + clear), then copy_init.
    _llk_math_welfords_sfpu_init_();
    ckernel::sfpu::_clear_previous_mean_and_m2_();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        4 /* num_faces */, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    std::uint32_t start_idx = 0;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            WELFORD_INPUT_DST_INDEX, formats.math, formats.math);

        const bool last = (tile + 1 == params.TILE_CNT);
        if (last && !(WELFORD_LAST_START_ROW == 0 && WELFORD_LAST_NUM_ROWS == 32))
        {
            _llk_math_welfords_sfpu_params_(
                ckernel::sfpu::_calculate_welfords_partial_tile_<WELFORD_LUT_SIZE>,
                WELFORD_INPUT_DST_INDEX,
                start_idx,
                WELFORD_LAST_START_ROW,
                WELFORD_LAST_NUM_ROWS,
                welford_reciprocal_lut);
            start_idx += WELFORD_LAST_NUM_ROWS;
        }
        else
        {
            _llk_math_welfords_sfpu_params_(
                ckernel::sfpu::_calculate_welfords_tile_<WELFORD_LUT_SIZE>, WELFORD_INPUT_DST_INDEX, start_idx, welford_reciprocal_lut);
            start_idx += 32;
        }
    }

    if constexpr (WELFORD_FINALIZE_RAW)
    {
        _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_, WELFORD_MEAN_DST_INDEX);
    }
    else
    {
        // scale_idx = N - 1 -> variance = M2 / N (population variance).
        _llk_math_welfords_sfpu_params_(
            ckernel::sfpu::_store_mean_var_to_dst_row_<WELFORD_LUT_SIZE>, WELFORD_MEAN_DST_INDEX, start_idx - 1, welford_reciprocal_lut);
    }

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * 4);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX + 1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

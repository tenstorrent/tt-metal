// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Functional driver for the Welford SFPU kernel: TILE_CNT tiles of 32 samples for 32 columns are folded into the
// running mean and M2, then the mean and the population variance are written into row 0 of DEST tiles 2 and 3 and
// packed as result tiles 0 and 1. WELFORD_RECIP_SIZE: N > 0 an N-entry table of 1 / (i + 1), 0 the no-table form.

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

// DEST tile 0 receives the input; the mean and the M2 / variance go to tiles 2 and 3.
static constexpr std::uint32_t WELFORD_INPUT_DST_INDEX = 0;
static constexpr std::uint32_t WELFORD_MEAN_DST_INDEX  = 2;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

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

// The reciprocal table the kernel reads through a reference. Empty when WELFORD_RECIP_SIZE is 0.
static std::array<std::uint32_t, WELFORD_RECIP_SIZE> reciprocal_lut;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    for (std::uint32_t i = 0; i < WELFORD_RECIP_SIZE; ++i)
    {
        const float reciprocal = 1.0f / static_cast<float>(i + 1);
        std::uint32_t bits;
        __builtin_memcpy(&bits, &reciprocal, sizeof(bits));
        reciprocal_lut[i] = bits;
    }

    // Copy input tile from SrcA into dst.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Welford init: the SFPU configuration, the address mode, the replay buffer; clear the running mean and M2.
    _llk_math_welfords_sfpu_init_();
    ckernel::sfpu::_clear_previous_mean_and_m2_();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        // Input into dst tile 0 (the unpacker writes it when unpack_to_dest is set).
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            WELFORD_INPUT_DST_INDEX, formats.math, formats.math);

        // Fold the 32 rows of the tile into the running statistics; the sample count so far is tile * 32.
        _llk_math_welfords_sfpu_params_(
            ckernel::sfpu::_calculate_welfords_tile_<WELFORD_RECIP_SIZE>, WELFORD_INPUT_DST_INDEX, tile * 32, reciprocal_lut);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }

    // Finalize: the mean and the population variance into row 0 of dst tiles 2 and 3.
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_row_<WELFORD_RECIP_SIZE>, WELFORD_MEAN_DST_INDEX, params.TILE_CNT * 32 - 1, reciprocal_lut);
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // The input sections produce nothing to pack; release each one as the math thread finishes it.
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }

    // The finalize section: the mean tile and the variance tile.
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX + 1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

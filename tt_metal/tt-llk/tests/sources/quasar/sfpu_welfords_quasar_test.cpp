// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // WELFORDS_*, IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en

// Welford's running per-column mean / variance on Quasar.
//
//   T0 unpack: stage TILE_CNT tiles of buffer_A from L1 into Dest (unpack-to-dest).
//   T1 math:   fold every tile's rows, in order, into the running state held in LREG4/LREG5
//              (the last tile only over [WELFORDS_START_ROW, +WELFORDS_NUM_ROWS) when
//              WELFORDS_PARTIAL_LAST_TILE). With WELFORDS_SAVE_RESTORE the state is saved to
//              Dest before tile WELFORDS_SAVE_AFTER_TILES, cleared, and restored. The finalize
//              writes mean to tile WELFORDS_FINAL_DST and variance to the tile after it.
//   T2 pack:   pack all TILE_CNT tiles; Python compares only the lanes the kernel wrote.

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);

    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);

    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_welfords.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// fp32 bit patterns of 1/(i+1); an empty array selects the kernel's RISC-V division fallback.
template <std::size_t SIZE>
constexpr std::array<std::uint32_t, SIZE> make_welfords_reciprocal_lut()
{
    std::array<std::uint32_t, SIZE> lut {};
    for (std::size_t i = 0; i < SIZE; ++i)
    {
        lut[i] = __builtin_bit_cast(std::uint32_t, 1.0f / static_cast<float>(i + 1));
    }
    return lut;
}

constexpr std::array<std::uint32_t, WELFORDS_RECIP_SIZE> WELFORDS_LUT = make_welfords_reciprocal_lut<WELFORDS_RECIP_SIZE>();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);

    _llk_math_eltwise_sfpu_init_();
    welfords_init();
    welfords_clear_previous_mean_and_m2();

    const std::uint32_t last_tile = params.TILE_CNT - 1;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        if constexpr (WELFORDS_SAVE_RESTORE)
        {
            if (tile == WELFORDS_SAVE_AFTER_TILES)
            {
                SFPU_UNARY_CALL(
                    dest_sync,
                    is_fp32_dest_acc_en,
                    welfords_store_mean_m2_to_dst,
                    (WELFORDS_STATE_GROUPED),
                    params.DST_INDEX + WELFORDS_STATE_DST,
                    VectorMode::RC_custom,
                    WELFORDS_STATE_GROUP_ID);
                welfords_clear_previous_mean_and_m2();
                SFPU_UNARY_CALL(
                    dest_sync,
                    is_fp32_dest_acc_en,
                    welfords_load_mean_m2_from_dst,
                    (WELFORDS_STATE_GROUPED),
                    params.DST_INDEX + WELFORDS_STATE_DST,
                    VectorMode::RC_custom,
                    WELFORDS_STATE_GROUP_ID);
            }
        }

        const std::uint32_t start_idx = tile * TILE_R_DIM;
        if (WELFORDS_PARTIAL_LAST_TILE && tile == last_tile)
        {
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_welfords,
                (true /* PARTIAL_TILE */, WELFORDS_RECIP_SIZE),
                params.DST_INDEX + tile,
                VectorMode::RC_custom,
                start_idx,
                WELFORDS_LUT,
                WELFORDS_START_ROW,
                WELFORDS_NUM_ROWS);
        }
        else
        {
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_welfords,
                (false /* PARTIAL_TILE */, WELFORDS_RECIP_SIZE),
                params.DST_INDEX + tile,
                VectorMode::RC_custom,
                start_idx,
                WELFORDS_LUT,
                0u /* start_row */,
                static_cast<std::uint32_t>(TILE_R_DIM) /* num_rows */);
        }
    }

    // variance = M2 / N over every processed row, so the divisor index is N - 1.
    const std::uint32_t rows_processed = last_tile * TILE_R_DIM + (WELFORDS_PARTIAL_LAST_TILE ? WELFORDS_NUM_ROWS : TILE_R_DIM);
    const std::uint32_t scale_idx      = rows_processed - 1;
    if constexpr (WELFORDS_FACE_LAYOUT)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            welfords_store_mean_var_to_dst,
            (WelfordsOutputLayout::Face, WELFORDS_FINAL_GROUPED, WELFORDS_RECIP_SIZE),
            params.DST_INDEX + WELFORDS_FINAL_DST,
            VectorMode::RC_custom,
            scale_idx,
            WELFORDS_LUT,
            WELFORDS_FINAL_GROUP_ID);
    }
    else
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            welfords_store_mean_var_to_dst,
            (WelfordsOutputLayout::Row, false /* GROUPED */, WELFORDS_RECIP_SIZE),
            params.DST_INDEX + WELFORDS_FINAL_DST,
            VectorMode::RC_custom,
            scale_idx,
            WELFORDS_LUT,
            0u /* group_id */);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_pack_(params.DST_INDEX, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// =====================================================================
// Embedded piecewise-polynomial LUT (generic_lut_activation style).
//
// Approximates sigmoid(x) on [-4, 4] with NUM_SEGMENTS=4 quadratic
// (POLY_DEGREE=2) segments. Layout matches the reference kernel:
//   [b0..bN  (NUM_SEGMENTS+1 boundaries),
//    then per segment (POLY_DEGREE+1) coeffs c0..cn]
// No range reduction (the NO-range-reduction path).
//
// The Python golden replicates these EXACT values so PCC isolates
// kernel correctness.
// =====================================================================
constexpr std::uint32_t POLY_DEGREE  = 2;
constexpr std::uint32_t NUM_SEGMENTS = 4;
constexpr std::uint32_t LUT_SIZE     = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1);

constexpr std::array<float, LUT_SIZE> LUT_DATA = {
    // boundaries b0..b4
    -4.0f, -2.0f, 0.0f, 2.0f, 4.0f,
    // seg0 coeffs: c0, c1, c2
    0.38296354f, 0.17515847f, 0.02109685f,
    // seg1
    0.50329190f, 0.27505103f, 0.04113654f,
    // seg2
    0.49670810f, 0.27505103f, -0.04113654f,
    // seg3
    0.61703646f, 0.17515847f, -0.02109685f,
};

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

    _llk_unpack_A_init_<BroadcastType::NONE, false /* is_fp32_dest_acc_en */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, FACE_R_DIM, TILE_NUM_FACES, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* is_fp32_dest_acc_en */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

using namespace ckernel;

namespace sfpi
{

// Generic polynomial evaluation using Horner's method (DEGREE == 2 path).
inline vFloat eval_polynomial_deg2(const float* coeffs, vFloat x)
{
    // Quadratic: y = c0 + c1*x + c2*x^2 -> Horner: (c2*x + c1)*x + c0
    return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
}

// Generic piecewise polynomial LUT (NO range reduction).
inline void piecewise_generic_lut()
{
    constexpr std::uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr std::uint32_t COEFF_OFFSET       = NUM_SEGMENTS + 1; // skip boundaries

    const auto& lut = LUT_DATA;

    for (int d = 0; d < 32; d++)
    {
        vFloat x = dst_reg[d];

        // Clamp x to [b0, bN] using branchless vec_min_max.
        vFloat min_bound = lut[0];
        vFloat max_bound = lut[NUM_SEGMENTS];
        vFloat x_clamped = x;
        vec_min_max(min_bound, x_clamped); // x_clamped = max(x, lut[0])
        vec_min_max(x_clamped, max_bound); // x_clamped = min(x_clamped, lut[NUM_SEGMENTS])

        // Start with segment 0, then override with the correct segment.
        vFloat result = eval_polynomial_deg2(&lut[COEFF_OFFSET], x_clamped);
        for (std::uint32_t seg = 1; seg < NUM_SEGMENTS; seg++)
        {
            v_if (x_clamped >= lut[seg])
            {
                result = eval_polynomial_deg2(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x_clamped);
            }
            v_endif;
        }

        dst_reg[d] = result;
    }
}

} // namespace sfpi

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Copy srcA to dest (A2D), then run the embedded piecewise-polynomial LUT
    // on dest via SFPU.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Configure the SFPU before issuing any SFPU instructions.
    ckernel::sfpu::_init_sfpu_config_reg();

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    for (std::uint32_t tile_num = 0; tile_num < params.TILE_CNT; ++tile_num)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile_num, formats.math, formats.math);

        _llk_math_eltwise_sfpu_start_(tile_num);
        sfpi::piecewise_generic_lut();
        _llk_math_eltwise_sfpu_done_();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile_num = 0; tile_num < params.TILE_CNT; ++tile_num)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_num, L1_ADDRESS(params.buffer_Res[tile_num]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

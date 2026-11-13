// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MXFP8 (E4M3 or E5M2) stored as remapped bfloat8_b, restored by an SFPU bit decode (mxfp8_decode_tile in
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mxfp8_decode.h).
//
// Unpack A (Bfp8_b) -> datacopy to Dest -> SFPU on Pack -> pack Float16_b.
// The unpacker normalizes the 7-bit magnitude m: the FP32 image is 2^(q+k) * (1 + w/64), w = mantissa bits
// 22..17. The decode keeps the top P bits of w (the MXFP8 mantissa: P = 3 for E4M3, 2 for E5M2), drops the
// remaining bits b (bits 19..17 or 20..17) and adds g(b) = 2b - popcount(b) to the exponent:
//     out = U - (b << 17) + (g(b) << 23)
// g depends on b only, so the decode works for any shared exponent (the MXFP8 scale lives in it).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

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
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "sfpi.h"

#include "ckernel_sfpu_mxfp8_decode.h"

namespace
{
#ifdef MXFP8_E5M2
constexpr bool kE5M2 = true;
#else
constexpr bool kE5M2 = false;
#endif
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    std::uint64_t decode_cycles = 0;
    std::uint64_t empty_cycles  = 0;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();

        _llk_math_eltwise_sfpu_start_(0 /* dst_index */);
        ckernel::tensix_sync();
        const std::uint64_t e0 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t e1 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t t0 = ckernel::read_wall_clock();
        ckernel::sfpu::mxfp8_decode_tile<kE5M2>();
        ckernel::tensix_sync();
        const std::uint64_t t1 = ckernel::read_wall_clock();
        decode_cycles += t1 - t0;
        empty_cycles += e1 - e0;
        _llk_math_eltwise_sfpu_done_();

        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
    ckernel::tensix_sync();
    volatile std::uint32_t* timing = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[params.TILE_CNT]);
    timing[0]                      = static_cast<std::uint32_t>(decode_cycles);
    timing[1]                      = static_cast<std::uint32_t>(empty_cycles);
    timing[2]                      = params.TILE_CNT;
    timing[3]                      = 0x4D584638; // "MXF8" marker
}

#endif

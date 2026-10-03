// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK SFPU quantization functional test kernel: quant, requant and dequant with a per-tensor scale, in the two forms
// of sources/sfpu_quant_scalar_perf.cpp (same QUANT_* configuration); tile A is the input, tile B the scale tile.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifndef QUANT_OP
#define QUANT_OP 0
#endif
#ifndef QUANT_SCALE_FORM
#define QUANT_SCALE_FORM 0
#endif
#ifndef QUANT_ZP_BITS
#define QUANT_ZP_BITS 0x40400000u
#endif
#ifndef QUANT_SCALE_BITS
#define QUANT_SCALE_BITS 0x3F000000u
#endif

static constexpr std::uint32_t QUANT_NEG_ZP_BITS = QUANT_ZP_BITS ^ 0x80000000u;

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
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    if constexpr (QUANT_SCALE_FORM == 0)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_quant.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr bool SCALAR = (QUANT_SCALE_FORM == 1);

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::quant_int32>();
        if constexpr (SCALAR)
        {
            sfpu::quant_init_scalar_scale<false, false, DataFormat::Int32>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
        }
        else
        {
            sfpu::quant_init<false, false, DataFormat::Int32>(QUANT_ZP_BITS);
        }
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::requant_int32>();
        if constexpr (SCALAR)
        {
            sfpu::requant_init_scalar_scale<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
        }
        else
        {
            sfpu::requant_init<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS);
        }
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::dequant_int32>();
        if constexpr (SCALAR)
        {
            sfpu::dequant_init_scalar_scale<false, false, false>(QUANT_NEG_ZP_BITS, QUANT_SCALE_BITS);
        }
        else
        {
            sfpu::dequant_init<false, false, false>(QUANT_NEG_ZP_BITS);
        }
    }

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
    if constexpr (!SCALAR)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(1, formats.math, formats.math);
    }
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_params_(sfpu::calculate_quant_int32<false, 8, false, SCALAR>, 0, 1, 0, VectorMode::RC);
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_params_(sfpu::calculate_requant_int32<false, 8, false, false, SCALAR>, 0, 1, 0, VectorMode::RC);
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_params_(sfpu::calculate_dequant_int32<false, 8, false, false, SCALAR>, 0, 1, 0, VectorMode::RC);
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
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

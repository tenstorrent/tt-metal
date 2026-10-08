// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// IEEE-754 coverage for the tt-llk ordered scalar compares (tt-llk#1701 item 3).
//
// Drives _calculate_comp_unary_<APPROX, unary_gt|lt|ge|le> over one tile of special values. The threshold is
// the raw fp32 bit pattern in SFPU_UNARY_SCALAR and reaches the kernel through one of two entry points
// (COMP_SCALAR_VIA_UINT32):
//   - the vFloat overload, decoded with Converter::as_float, so the test can pin -0.0, +-inf and +-NaN
//     thresholds independently of how the uint32 overload decodes its argument;
//   - the shipped std::uint32_t entry point, so the same compares are also checked through the public API.
// A header without the vFloat overload (before the fix) still compiles this driver: the vFloat variants then
// fall back to the uint32 entry point, whose numeric decode of the bits only agrees with the +0.0 threshold.

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// The tile the datacopy writes and the SFPU then reads; both calls must use the same index.
static constexpr std::uint32_t DST_INDEX = 0;

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

    _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace ckernel;

// True when the header has the vFloat overload of _calculate_comp_unary_ (added with the item 3 fix).
template <typename T, typename = void>
struct has_vfloat_comp_unary : std::false_type
{
};

template <typename T>
struct has_vfloat_comp_unary<
    T,
    std::void_t<decltype(ckernel::sfpu::_calculate_comp_unary_<APPROX_MODE, SFPU_UNARY_OPERATION, 8 /* ITERATIONS */>(std::declval<T>()))>>
    : std::true_type
{
};

// Called only when the overload exists; the dependent argument keeps the call out of the base header's build.
template <typename T>
sfpi_inline void comp_unary_vfloat(T threshold)
{
    ckernel::sfpu::_calculate_comp_unary_<APPROX_MODE, SFPU_UNARY_OPERATION, 8 /* ITERATIONS */>(threshold);
}

// A template, so the branch not taken is discarded rather than checked against the header in use.
template <bool VIA_UINT32>
sfpi_inline void run_comp_unary()
{
    if constexpr (VIA_UINT32 || !has_vfloat_comp_unary<sfpi::vFloat>::value)
    {
        ckernel::sfpu::_calculate_comp_unary_<APPROX_MODE, SFPU_UNARY_OPERATION, 8 /* ITERATIONS */>(SFPU_UNARY_SCALAR);
    }
    else
    {
        comp_unary_vfloat(sfpi::vFloat(ckernel::sfpu::Converter::as_float(SFPU_UNARY_SCALAR)));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // The compares need only the generic SFPU init (config reg, ADDR_MOD_7, dest RWC reset).
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        DST_INDEX, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // ITERATIONS=8 per face at VectorMode::RC covers the whole tile.
    _llk_math_eltwise_unary_sfpu_params_(
        [] { run_comp_unary<COMP_SCALAR_VIA_UINT32>(); },
        DST_INDEX,
        VECTOR_MODE);

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
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
